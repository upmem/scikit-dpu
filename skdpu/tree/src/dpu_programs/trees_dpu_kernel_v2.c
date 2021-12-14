/**
 * @file trees_dpu_kernel_v2.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @author Julien Legriel (jlegriel@upmem.com)
 * @brief DPU side of the tree algorithm
 *
 */

#ifndef _TREES_DPU_KERNEL_V2_H_
#define _TREES_DPU_KERNEL_V2_H_ /**< guard to prevent linking with CPU binaries   \
                              */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <mutex.h>

#include "../trees_common.h"

/*------------------ INPUT ------------------------------*/
/** @name Host
 * Variables for host application communication
 */
/**@{*/
__host size_t n_points;
__host size_t n_features;
__host size_t n_classes;
/**@}*/

/**
 * @brief array of points features
 * The features are stored consecutively in MRAM:
 * <point1 feature1><point2 feature 1>....<pointn feature1><point1 feature2> ...
 **/
__mram_noinit feature_t t_features[MAX_FEATURE_DPU]; /**< Vector of features. */
/**
 * @brief array of classes
 **/
__mram_noinit uint8_t t_targets[MAX_SAMPLES_DPU];     /**< Vector of targets. */

/**
 * @brief an array of commands to execute
 * Note: there is no check that the host does not send two commands for the same leaf
 * However this would be undefined behavior if for instance some tasklets are in parallel
 * doing a SPLIT_EVALUATE and SPLIT_COMMIT for the same leaf
 **/
__host struct Command cmds_array[MAX_NB_LEAF];
/**
 * @brief number of commands in the array
 **/
__host uint16_t nb_cmds;

/**
 * @brief array to store the scores or gini count for each leaf and classes
 **/
__mram uint32_t gini_cnt[MAX_NB_LEAF * MAX_CLASSES];

// Les points d’une feuille de l’arbre sont stockés consécutivement en MRAM
// entre les index leaf_start_index et leaf_end_index
// should be OK to keep this in WRAM, with MAX_NB_LEAF=1000, it is another 8K

/**
 * @brief the points in one tree leaf are stored consecutively in MRAM
 * This order is maitained after a split commit by reordering the t_features and the t_targets 
 * arrays in MRAM
 * leaf_start_index is the index of the first point in the tree leaf
 * leaf_end_index is the end index of the tree leaf, the start index of the next leaf
 **/ 
__host uint32_t n_leaves;
__host uint32_t leaf_start_index[MAX_NB_LEAF];
__host uint32_t leaf_end_index[MAX_NB_LEAF]; // this is the first index of next tree leaf

#define SIZE_BATCH 64
#define SIZE_COMMIT_BATCH 256

// WRAM buffer for feature values, SIZE_BATCH per tasklet
// Note: using a bigger buffer to accomodate also the commit command
__dma_aligned feature_t feature_values[NR_TASKLETS][SIZE_BATCH + 16];
__dma_aligned feature_t feature_values_commit_low[NR_TASKLETS][SIZE_COMMIT_BATCH];
__dma_aligned feature_t feature_values_commit_high[NR_TASKLETS][SIZE_COMMIT_BATCH];

// WRAM buffer for classes values, SIZE_BATCH values per tasklet
__dma_aligned uint8_t classes_values[NR_TASKLETS][SIZE_BATCH + 16];
// WRAM buffer for the gini count
uint32_t w_gini_cnt[NR_TASKLETS][MAX_CLASSES];
__dma_aligned uint32_t c_gini_cnt[NR_TASKLETS][MAX_CLASSES + 16];
MUTEX_INIT(gini_mutex);

// batch counter and mutex
uint32_t batch_cnt = 0;
uint32_t cmd_cnt = 0;
MUTEX_INIT(batch_mutex);

/**
 * @brief each tasklet gets a next batch to execute when it is ready
 * Each command is divided in a certain number of batches
 * For a SPLIT_EVALUATE command, a batch is a subset of the points of the tree leaf
 * for which the gini counts need to be computed
 *
 **/
static bool get_next_command(uint16_t * index_cmd, uint32_t * index_batch) {

    cmd_cnt++;
    *index_cmd = cmd_cnt;
    batch_cnt = 1;
    if(cmd_cnt >= nb_cmds) return false;
    if(cmds_array[cmd_cnt].type == SPLIT_EVALUATE) {
        uint16_t leaf_index = cmds_array[cmd_cnt].leaf_index;
        *index_batch = leaf_start_index[leaf_index];
    }
    else {
        // commit case
        *index_batch = 0; 
    }
    return true;
}

static bool get_next_batch_evaluate(uint16_t * index_cmd, uint32_t * index_batch) {

    uint16_t leaf_index = cmds_array[cmd_cnt].leaf_index;
    *index_batch = batch_cnt * SIZE_BATCH + leaf_start_index[leaf_index];
    if(*index_batch >= leaf_end_index[leaf_index]) {
        // finished processing this command, go to next
        return get_next_command(index_cmd, index_batch);
    }
    *index_cmd = cmd_cnt;
    ++batch_cnt;
    return true;
}

static bool get_next_batch_commit(uint16_t * index_cmd, uint32_t * index_batch) {

    if(batch_cnt >= n_features)
        return get_next_command(index_cmd, index_batch);

    *index_cmd = cmd_cnt;
    *index_batch = batch_cnt;
    if(*index_batch >= cmds_array[cmd_cnt].feature_index) {

        // The trick here is, the feature that has the split needs to be updated last
        // Otherwise we loose the information how to order for the other features
        // for the commit, the batch count is the next feature to handle
        (*index_batch)++;
        if(*index_batch == n_features)
            *index_batch = cmds_array[cmd_cnt].feature_index;
    }
    batch_cnt++;
    return true;
}

static bool get_next_batch(uint16_t * index_cmd, uint32_t * index_batch) {

    mutex_lock(batch_mutex);
    bool res = false;
    if(cmd_cnt >= nb_cmds) res = false;
    else if(cmds_array[cmd_cnt].type == SPLIT_EVALUATE) {
        res = get_next_batch_evaluate(index_cmd, index_batch);
    }
    else {
        // commit
        // no next batch, go to next command
        res = get_next_batch_commit(index_cmd, index_batch);
    }
    mutex_unlock(batch_mutex);
    return res;
}

/**
 * @brief load features from MRAM to WRAM for a batch, while taking care of alignment constraints
 **/
static feature_t* load_feature_values(
        uint32_t index_batch, 
        uint16_t feature_index, 
        uint16_t size_batch) {

    // Need to handle the constraint that MRAM read must be on a MRAM address aligned on 8 bytes
    // with a size also aligned on 8 bytes
    uint32_t start_index = feature_index * n_points + index_batch;
    uint32_t start_index_8align = (((start_index * sizeof(feature_t)) >> 3) << 3) / sizeof(feature_t);
    uint32_t diff = start_index - start_index_8align;
    uint32_t size_batch_8align = (((((size_batch + diff) * sizeof(feature_t)) + 7) >> 3) << 3) / sizeof(feature_t);

    mram_read(&t_features[start_index_8align], feature_values[me()], size_batch_8align * sizeof(feature_t));

    return feature_values[me()] + diff;
}

/**
 * @brief load classes values from MRAM to WRAM for a batch, while taking care of alignment constraints
 **/
static uint8_t* load_classes_values(uint32_t index_batch, uint16_t size_batch) {

    // Need to handle the constraint that MRAM read must be on a MRAM address aligned on 8 bytes
    // with a size also aligned on 8 bytes
    uint32_t start_index = index_batch;
    uint32_t start_index_8align = (start_index >> 3) << 3;
    uint32_t diff = start_index - start_index_8align;
    uint32_t size_batch_8align = ((size_batch + diff + 7) >> 3) << 3;

    mram_read(&t_targets[start_index_8align], classes_values[me()], size_batch_8align);

    return classes_values[me()] + diff; 
}

/**
 * @brief update the gini counts in MRAM for a given leaf
 **/
static void update_gini_cnt(uint16_t leaf_index, uint32_t * gini_cnt_input) {

    mutex_lock(gini_mutex);
    // load the current gini cnt
    // Need to take into account the MRAM alignement constraints
    printf("update gini count %u leaf %u\n", me(), leaf_index);
    uint32_t start_index = leaf_index * n_classes;
    uint32_t start_index_8align = (((start_index * sizeof(uint32_t)) >> 3) << 3) / sizeof(uint32_t);
    uint32_t diff = start_index - start_index_8align;
    uint32_t size_8align = (((((n_classes + diff) * sizeof(uint32_t)) + 7) >> 3) << 3) / sizeof(uint32_t);
    mram_read(&gini_cnt[start_index_8align], c_gini_cnt[me()], size_8align * sizeof(uint32_t));
    for(int i = 0; i < n_classes; ++i) {
        printf("class %u current count %u=%u  incr %u\n", i, 
                c_gini_cnt[me()][diff + i], gini_cnt[start_index_8align + i], gini_cnt_input[i]);
        c_gini_cnt[me()][diff + i] += gini_cnt_input[i];
    }
    // store the updated values
    mram_write(c_gini_cnt[me()], &gini_cnt[start_index_8align], size_8align * sizeof(uint32_t));
    mutex_unlock(gini_mutex);
}

/**
 * @brief handle a batch for the SPLIT_EVALUATE command
 **/
static void do_split_evaluate(uint16_t index_cmd, uint32_t index_batch) {

    printf("do_split_evaluate %d cmd %u batch %u\n", me(), index_cmd, index_batch);
    uint16_t size_batch = SIZE_BATCH;
    if(index_batch + SIZE_BATCH > leaf_end_index[cmds_array[index_cmd].leaf_index])
        size_batch = leaf_end_index[cmds_array[index_cmd].leaf_index] - index_batch;

    feature_t* features_val = load_feature_values(index_batch, cmds_array[index_cmd].feature_index, size_batch);
    uint8_t* classes_val = load_classes_values(index_batch, size_batch);

    memset(w_gini_cnt[me()], 0, n_classes * sizeof(uint32_t));

    for(int i = 0; i < size_batch; ++i) {

        if(features_val[i] <= cmds_array[index_cmd].feature_threshold) {

            // increment the gini count for this class
            printf("tid %u point %u class %u %u feature %f threshold %f index_batch %u size_batch %u\n", 
                    me(), index_batch + i, classes_val[i], t_targets[index_batch + i], features_val[i], cmds_array[index_cmd].feature_threshold, 
                    index_batch, size_batch);
            w_gini_cnt[me()][classes_val[i]]++;
        }
    }

    // update the gini count in MRAM
    update_gini_cnt(cmds_array[index_cmd].leaf_index, w_gini_cnt[me()]);
    printf("do_split_evaluate end %d cmd %u batch %u\n", me(), index_cmd, index_batch);
}

// three possibilities
// lower buffer is completed
// higher buffer is completed
// both are completed
enum swap_status {
    low,
    high,
    both
};

static enum swap_status swap_buffers(
        feature_t * b_low, 
        feature_t * b_high, 
        feature_t * b_cmp_low, 
        feature_t * b_cmp_high, 
        uint16_t sz, 
        //TODO may have different size for the two buffers
        feature_t threshold) {

    uint16_t low = 0, high = 0;
    while(low < sz && high < sz) {

        bool incr = false;
        if(b_cmp_low[low] <= threshold) {
            b_low++;
            incr = true;
        }
        if(b_cmp_high[high] > threshold) {
            b_high++;
            incr = true;
        }
        if(!incr) {
            // swap
            feature_t swap = b_low[low];
            b_low[low] = b_high[high];
            b_high[high] = swap;
        }
    }
    while(low < sz && b_cmp_low[low] <= threshold)
        low++;
    while(high < sz && b_cmp_high[high] > threshold)
        high++;

    //TODO if there is never any swap, no need to write back

    if(low >= sz && high >= sz)
        return both;
    if(low >= sz)
        return low;
    else
        return high;
}

//TODO: what about the alignment of MRAM transferts ?
//When we load points in a specific leaf node, what if the point is not aligned on 8 byte (a feature is only 4 byte)
static void do_split_commit(uint16_t index_cmd, uint32_t feature_index) {

#if 0
    // TODO for the commit, we could pipeline the mram_write with the swap

    // initialization, load buffers
    uint32_t size_batch_low = SIZE_COMMIT_BATCH; // TODO
    uint32_t size_batch_high = SIZE_COMMIT_BATCH;
    uint16_t leaf_index = cmds_array[index_cmd].leaf_index;
    uint16_t cmp_feature_index = cmds_array[index_cmd].feature_index;

    // load buffers for the feature we want to handle the swap for
    load_feature_values(leaf_start_index[leaf_index], feature_index, 
            size_batch_low, feature_values_commit_low[me()]);
    load_feature_values(leaf_end_index[leaf_index] - size_batch_high, feature_index, 
            size_batch_high, feature_values_commit_high[me()]);

    // load buffers for the split feature, the one used for comparison
    load_feature_values(leaf_start_index[leaf_index], cmp_feature_index, 
            size_batch_low, feature_values_commit_cmp_low[me()]);
    load_feature_values(leaf_end_index[leaf_index] - size_batch_high, cmp_feature_index, 
            size_batch_high, feature_values_commit_cmp_high[me()]);

    bool commit_loop = true;
    while(commit_loop) {

        enum swap_status status = 
            swap_buffers(feature_values_commit_low[me()], feature_values_commit_high[me()], 
                feature_values_commit_cmp_low[me()], feature_values_commit_cmp_high[me()], 
                sz, cmds_array[index_cmd].feature_threshold);

        if(status == both) {

            // write both buffers back and get new ones
            // If no new buffer, this is the end, done
            // If only one buffer, it needs to be loaded and sorted individually and write back
        }
        else if(status == low) {

            // write low buffer back and get next one
            // if no next buffer, this is the end
            // reorder the high buffer correctly and write it back
        }
        else if (status == high) {

            // write high buffer back and get next one
            // if no next buffer, this is the end
            // reorder the low buffer correctly and write it back
        }
    }
#endif
}

BARRIER_INIT(barrier, NR_TASKLETS);

/*================== MAIN FUNCTION ======================*/
/**
 * @brief Main function DPU side.
 *
 * @return int 0 on success.
 */
int main() {

    //////////////// test
    // set points/classes
    if(me() == 0) {
        int k = 0;
        n_points = 200;
        n_features = 5;
        for(int j = 0; j < n_features; ++j) {
            for(int i = 0; i < n_points; ++i) {
                if(i < n_points * 0.3)
                    t_features[k] = 10;
                else 
                    t_features[k] = 50;
                k++;
            }
        }
        n_classes = 4;
        for(int i = 0; i < n_points; ++i) {
            t_targets[i] = i & 3;
            printf("point %u target %u\n", i, t_targets[i]);
        }
        nb_cmds = 2;
        struct Command cmd1, cmd2;
        cmd1.type = 0;
        cmd1.feature_index = 0;
        cmd1.feature_threshold = 30;
        cmd1.leaf_index = 0;
        cmd2.type = 0;
        cmd2.feature_index = 2;
        cmd2.feature_threshold = 60;
        cmd2.leaf_index = 1;
        cmds_array[0] = cmd1;
        cmds_array[1] = cmd2;
        n_leaves = 2;
        leaf_start_index[0] = 0;
        leaf_end_index[0] = n_points >> 1;
        leaf_start_index[1] = n_points >> 1;
        leaf_end_index[1] = n_points;
    }
    /////////////////////

    barrier_wait(&barrier);

    uint16_t index_cmd = 0; 
    uint32_t index_batch = 0;

    while(get_next_batch(&index_cmd, &index_batch)) {

        if(cmds_array[index_cmd].type == SPLIT_EVALUATE) {
            do_split_evaluate(index_cmd, index_batch);
        }
        else if(cmds_array[index_cmd].type == SPLIT_COMMIT) {
            /*do_split_commit(index_cmd, index_batch);*/
        }
        else {
            // TODO error
        }
    }

    barrier_wait(&barrier);

    //////////////// test
    if(me() == 0) {
        for(int l = 0; l < n_leaves; ++l) {
            uint32_t tmp_gini_cnt[MAX_CLASSES];
            mram_read(&gini_cnt[l * n_classes], tmp_gini_cnt, n_classes * sizeof(uint32_t));
            printf("leaf %u:\n", l);
            for(int i = 0; i < n_classes; ++i)
                printf("gini count class %u: %u/%u\n", i, tmp_gini_cnt[i], n_points);
        }
    }
    /////////////////////

    return 0;
}

#endif
