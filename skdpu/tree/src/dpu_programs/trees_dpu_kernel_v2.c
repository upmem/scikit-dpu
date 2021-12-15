/**
 * @file trees_dpu_kernel_v2.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @author Julien Legriel (jlegriel@upmem.com)
 * @brief DPU side of the tree algorithm
 *
 */

#ifndef _TREES_DPU_KERNEL_V2_H_
#define _TREES_DPU_KERNEL_V2_H_ /**< guard to prevent linking with CPU         \
                                 * binaries                                    \
                                 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <assert.h>

#include "../trees_common.h"

#define DEBUG

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
__mram_noinit uint8_t t_targets[MAX_SAMPLES_DPU]; /**< Vector of targets. */

/**
 * @brief an array of commands to execute
 * Note: there is no check that the host does not send two commands for the same
 *leaf However this would be undefined behavior if for instance some tasklets
 *are in parallel doing a SPLIT_EVALUATE and SPLIT_COMMIT for the same leaf
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

/**
 * @brief the points in one tree leaf are stored consecutively in MRAM
 * This order is maitained after a split commit by reordering the t_features and
 *the t_targets arrays in MRAM leaf_start_index is the index of the first point
 *in the tree leaf leaf_end_index is the end index of the tree leaf, the start
 *index of the next leaf
 **/
__host uint32_t n_leaves;
__host uint32_t leaf_start_index[MAX_NB_LEAF];
__host uint32_t
    leaf_end_index[MAX_NB_LEAF]; // this is the first index of next tree leaf

/**
 * @brief size of batch of features read at once for the SPLIT_EVALUATE and
 *SPLIT_COMMIT commands
 **/
#define SIZE_BATCH 32

/**
 * WRAM buffers for feature values, SIZE_BATCH per tasklet
 * Note: using a slightly bigger buffer to accomodate alignment to 8 bytes
 * */
__dma_aligned feature_t feature_values[NR_TASKLETS][SIZE_BATCH + 16];
__dma_aligned feature_t feature_values2[NR_TASKLETS][SIZE_BATCH];
__dma_aligned feature_t feature_values3[NR_TASKLETS][SIZE_BATCH];
__dma_aligned feature_t feature_values4[NR_TASKLETS][SIZE_BATCH];

/**
 * @ brief WRAM buffer for classes values, SIZE_BATCH values per tasklet
 **/
__dma_aligned uint8_t classes_values[NR_TASKLETS][SIZE_BATCH + 16];

/**
 * @brief WRAM buffer for the gini count
 **/
uint32_t w_gini_cnt[NR_TASKLETS][MAX_CLASSES];
__dma_aligned uint32_t c_gini_cnt[NR_TASKLETS][MAX_CLASSES + 16];
MUTEX_INIT(gini_mutex);

/**
 * @brief batch counter and mutex
 **/
uint32_t batch_cnt = 0;
uint32_t cmd_cnt = 0;
MUTEX_INIT(batch_mutex);

/**
 * @brief split commit synchronization
 **/
MUTEX_INIT(commit_mutex);
uint8_t cmd_tasklet_cnt[MAX_NB_LEAF] = {0};

/**
 * @brief each tasklet gets a next batch to execute when it is ready
 * Each command is divided in a certain number of batches
 * For a SPLIT_EVALUATE command, a batch is a subset of the points of the tree
 * leaf for which the gini counts need to be computed
 **/
static bool get_next_command(uint16_t *index_cmd, uint32_t *index_batch) {

  cmd_cnt++;
  *index_cmd = cmd_cnt;
  batch_cnt = 1;
  if (cmd_cnt >= nb_cmds)
    return false;
  if (cmds_array[cmd_cnt].type == SPLIT_EVALUATE) {
    uint16_t leaf_index = cmds_array[cmd_cnt].leaf_index;
    *index_batch = leaf_start_index[leaf_index];
  } else {
    // commit case
    *index_batch = 0;
  }
  return true;
}

static bool get_next_batch_evaluate(uint16_t *index_cmd,
                                    uint32_t *index_batch) {

  uint16_t leaf_index = cmds_array[cmd_cnt].leaf_index;
  *index_batch = batch_cnt * SIZE_BATCH + leaf_start_index[leaf_index];
  if (*index_batch >= leaf_end_index[leaf_index]) {
    // finished processing this command, go to next
    return get_next_command(index_cmd, index_batch);
  }
  *index_cmd = cmd_cnt;
  ++batch_cnt;
  return true;
}

static bool get_next_batch_commit(uint16_t *index_cmd, uint32_t *index_batch) {

  // The trick here is, the feature that has the split needs to be updated last
  // Otherwise we loose the information how to order for the other features
  // for the commit, the batch count is the next feature to handle
  if (batch_cnt == cmds_array[cmd_cnt].feature_index)
    batch_cnt++;

  if (batch_cnt >= n_features)
    return get_next_command(index_cmd, index_batch);

  *index_cmd = cmd_cnt;
  *index_batch = batch_cnt;
  batch_cnt++;
  return true;
}

static bool get_next_batch(uint16_t *index_cmd, uint32_t *index_batch) {

  mutex_lock(batch_mutex);
  bool res = false;
  if (cmd_cnt >= nb_cmds)
    res = false;
  else if (cmds_array[cmd_cnt].type == SPLIT_EVALUATE) {
    res = get_next_batch_evaluate(index_cmd, index_batch);
  } else {
    // commit
    // no next batch, go to next command
    res = get_next_batch_commit(index_cmd, index_batch);
  }
  mutex_unlock(batch_mutex);
  return res;
}

/**
 * @brief load features from MRAM to WRAM for a batch, while taking care of
 *alignment constraints
 **/
static feature_t *load_feature_values(uint32_t index_batch,
                                      uint16_t feature_index,
                                      uint16_t size_batch,
                                      feature_t *feature_values) {

  // Need to handle the constraint that MRAM read must be on a MRAM address
  // aligned on 8 bytes with a size also aligned on 8 bytes
  uint32_t start_index = feature_index * n_points + index_batch;
  uint32_t start_index_8align =
      (((start_index * sizeof(feature_t)) >> 3) << 3) / sizeof(feature_t);
  uint32_t diff = start_index - start_index_8align;
  uint32_t size_batch_8align =
      (((((size_batch + diff) * sizeof(feature_t)) + 7) >> 3) << 3) /
      sizeof(feature_t);

  mram_read(&t_features[start_index_8align], feature_values,
            size_batch_8align * sizeof(feature_t));

  return feature_values + diff;
}

/**
 * @brief load classes values from MRAM to WRAM for a batch, while taking care
 *of alignment constraints
 **/
static uint8_t *load_classes_values(uint32_t index_batch, uint16_t size_batch) {

  // Need to handle the constraint that MRAM read must be on a MRAM address
  // aligned on 8 bytes with a size also aligned on 8 bytes
  uint32_t start_index = index_batch;
  uint32_t start_index_8align = (start_index >> 3) << 3;
  uint32_t diff = start_index - start_index_8align;
  uint32_t size_batch_8align = ((size_batch + diff + 7) >> 3) << 3;

  mram_read(&t_targets[start_index_8align], classes_values[me()],
            size_batch_8align);

  return classes_values[me()] + diff;
}

/**
 * @brief update the gini counts in MRAM for a given leaf
 **/
static void update_gini_cnt(uint16_t leaf_index, uint32_t *gini_cnt_input) {

  mutex_lock(gini_mutex);
  // load the current gini cnt
  // Need to take into account the MRAM alignement constraints
#ifdef DEBUG
  printf("update gini count %u leaf %u\n", me(), leaf_index);
#endif
  uint32_t start_index = leaf_index * n_classes;
  uint32_t start_index_8align =
      (((start_index * sizeof(uint32_t)) >> 3) << 3) / sizeof(uint32_t);
  uint32_t diff = start_index - start_index_8align;
  uint32_t size_8align =
      (((((n_classes + diff) * sizeof(uint32_t)) + 7) >> 3) << 3) /
      sizeof(uint32_t);
  mram_read(&gini_cnt[start_index_8align], c_gini_cnt[me()],
            size_8align * sizeof(uint32_t));
  for (int i = 0; i < n_classes; ++i) {
#ifdef DEBUG
    printf("class %u current count %u=%u  incr %u\n", i,
           c_gini_cnt[me()][diff + i], gini_cnt[start_index_8align + i],
           gini_cnt_input[i]);
#endif
    c_gini_cnt[me()][diff + i] += gini_cnt_input[i];
  }
  // store the updated values
  mram_write(c_gini_cnt[me()], &gini_cnt[start_index_8align],
             size_8align * sizeof(uint32_t));
  mutex_unlock(gini_mutex);
}

/**
 * @brief handle a batch for the SPLIT_EVALUATE command
 **/
static void do_split_evaluate(uint16_t index_cmd, uint32_t index_batch) {

#ifdef DEBUG
  printf("do_split_evaluate %d cmd %u batch %u\n", me(), index_cmd,
         index_batch);
#endif
  uint16_t size_batch = SIZE_BATCH;
  if (index_batch + SIZE_BATCH >
      leaf_end_index[cmds_array[index_cmd].leaf_index])
    size_batch = leaf_end_index[cmds_array[index_cmd].leaf_index] - index_batch;

  feature_t *features_val =
      load_feature_values(index_batch, cmds_array[index_cmd].feature_index,
                          size_batch, feature_values[me()]);
  uint8_t *classes_val = load_classes_values(index_batch, size_batch);

  memset(w_gini_cnt[me()], 0, n_classes * sizeof(uint32_t));

  for (int i = 0; i < size_batch; ++i) {

    if (features_val[i] <= cmds_array[index_cmd].feature_threshold) {

      // increment the gini count for this class
#ifdef DEBUG
      printf("tid %u point %u class %u %u feature %f threshold %f index_batch "
             "%u size_batch %u\n",
             me(), index_batch + i, classes_val[i], t_targets[index_batch + i],
             features_val[i], cmds_array[index_cmd].feature_threshold,
             index_batch, size_batch);
#endif
      w_gini_cnt[me()][classes_val[i]]++;
    }
  }

  // update the gini count in MRAM
  update_gini_cnt(cmds_array[index_cmd].leaf_index, w_gini_cnt[me()]);
#ifdef DEBUG
  printf("do_split_evaluate end %d cmd %u batch %u\n", me(), index_cmd,
         index_batch);
#endif
}

/**
 * @brief swap two feature values
 **/
static void swap(feature_t *a, feature_t *b) {

  feature_t swap = *a;
  *a = *b;
  *b = swap;
}

/** 
 * @brief three possibilities after the swap
 * lower buffer is completed
 * higher buffer is completed
 * both are completed
 **/
enum swap_status {
  LOW_SWAP_COMPLETED,
  HIGH_SWAP_COMPLETED,
  BOTH_SWAP_COMPLETED
};

static enum swap_status swap_buffers(feature_t *b_low, feature_t *b_high,
                                     feature_t *b_cmp_low,
                                     feature_t *b_cmp_high, uint16_t sz,
                                     feature_t threshold, bool *has_swap) {

  *has_swap = false;
  uint16_t low = 0, high = 0;
  bool self = b_cmp_low == b_low;
  assert(!self || b_cmp_high == b_high);
  while (low < sz && high < sz) {

    bool incr = false;
    if (b_cmp_low[low] <= threshold) {
      low++;
      incr = true;
    }
    if (b_cmp_high[high] > threshold) {
      high++;
      incr = true;
    }
    if (!incr) {
      // swap
      swap(b_low + low, b_high + high);
      // if the comparison buffer is the same
      // dont swap twice
      if (!self) {
        swap(b_cmp_low + low, b_cmp_high + high);
      }
      *has_swap = true;
    }
  }
  while (low < sz && b_cmp_low[low] <= threshold)
    low++;
  while (high < sz && b_cmp_high[high] > threshold)
    high++;

  if (low >= sz && high >= sz)
    return BOTH_SWAP_COMPLETED;
  if (low >= sz)
    return LOW_SWAP_COMPLETED;
  else
    return HIGH_SWAP_COMPLETED;
}

static void get_index_and_size_for_commit(uint16_t index_cmd, uint32_t *index,
                                          uint32_t *size) {

  uint16_t leaf_index = cmds_array[index_cmd].leaf_index;
  uint32_t start_index = leaf_start_index[leaf_index];
  *index =
      (((start_index * sizeof(feature_t) + 7) >> 3) << 3) / sizeof(feature_t);
  uint32_t nb_buffers = (leaf_end_index[leaf_index] - *index) / SIZE_BATCH;
  *size = nb_buffers * SIZE_BATCH;
}

static void store_features_values(uint32_t start_index, uint32_t feature_index,
                                  uint32_t size_batch,
                                  feature_t *feature_values) {

  // TODO if feature_index * n_points * sizeof(feature_t) is not aligned on 8
  // byte there is an issue
  // TODO do a macro to check the alignment constraints
  mram_write(feature_values,
             &t_features[feature_index * n_points + start_index],
             size_batch * sizeof(feature_t));
}

static int partition_buffer(feature_t *feature_values,
                            feature_t *feature_values_cmp, uint32_t size_batch,
                            feature_t feature_threshold) {

  int pivot = -1;
  assert(size_batch);
  for (int j = 0; j < size_batch; j++) {
    if (feature_values_cmp[j] < feature_threshold) {
      pivot++;
      if (pivot != j) {
        swap(feature_values + pivot, feature_values + j);
      }
    }
  }
  // TODO Not sure this loop is necessary
  while (feature_values_cmp[pivot] < feature_threshold)
    pivot++;

  return pivot;
}

// TODO break into smaller functions
static void do_split_commit(uint16_t index_cmd, uint32_t feature_index) {

  // TODO for the commit, we could pipeline the mram_write with the swap

#ifdef DEBUG
  printf("do_split_commit tid %d index_cmd %u feature_index %u\n", me(),
         index_cmd, feature_index);
#endif

  uint32_t start_index = 0, size = 0;

  // the main algorithm will work on aligned indexes, this means
  // the first index is aligned on 8 bytes, and the size is aligned on
  // SIZE_BATCH In the end we handle the prolog and epilog
  get_index_and_size_for_commit(index_cmd, &start_index, &size);
  uint16_t leaf_index = cmds_array[index_cmd].leaf_index;
  uint16_t cmp_feature_index = cmds_array[index_cmd].feature_index;
  bool self = feature_index == cmp_feature_index;

  bool commit_loop = true;
  bool load_low = true, load_high = true;
  uint32_t start_index_low = start_index;
  uint32_t start_index_high = start_index + size - SIZE_BATCH;
  feature_t *feature_values_commit_low, *feature_values_commit_cmp_low;
  feature_t *feature_values_commit_high, *feature_values_commit_cmp_high;
  int pivot;

  // only one buffer, directly partition it
  if (start_index_low == start_index_high) {

    feature_values_commit_low = load_feature_values(
        start_index_low, feature_index, SIZE_BATCH, feature_values[me()]);
    if (!self)
      feature_values_commit_cmp_low =
          load_feature_values(start_index_low, cmp_feature_index, SIZE_BATCH,
                              feature_values2[me()]);
    else
      feature_values_commit_cmp_low = feature_values_commit_low;

    pivot = partition_buffer(feature_values_commit_low,
                             feature_values_commit_cmp_low, SIZE_BATCH,
                             cmds_array[index_cmd].feature_threshold);
    store_features_values(start_index_low, feature_index, SIZE_BATCH,
                          feature_values_commit_low);

    commit_loop = false;
  }

  while (commit_loop) {

    // load buffers for the feature we want to handle the swap for
    // and load buffers of the split feature, used for the comparisons
    if (load_low) {
      feature_values_commit_low = load_feature_values(
          start_index_low, feature_index, SIZE_BATCH, feature_values[me()]);
      if (!self)
        feature_values_commit_cmp_low =
            load_feature_values(start_index_low, cmp_feature_index, SIZE_BATCH,
                                feature_values2[me()]);
      else
        feature_values_commit_cmp_low = feature_values_commit_low;
    }
    if (load_high) {
      feature_values_commit_high = load_feature_values(
          start_index_high, feature_index, SIZE_BATCH, feature_values3[me()]);
      if (!self)
        feature_values_commit_cmp_high =
            load_feature_values(start_index_high, cmp_feature_index, SIZE_BATCH,
                                feature_values4[me()]);
      else
        feature_values_commit_cmp_high = feature_values_commit_high;
    }

    bool has_swap;
    enum swap_status status = swap_buffers(
        feature_values_commit_low, feature_values_commit_high,
        feature_values_commit_cmp_low, feature_values_commit_cmp_high,
        SIZE_BATCH, cmds_array[index_cmd].feature_threshold, &has_swap);

    if (status == BOTH_SWAP_COMPLETED) {

      // write both buffers back and get new ones
      // If no new buffer, this is the end, done
      // If only one buffer, it needs to be loaded and sorted individually and
      // write back

      // no need to write if the buffer was just loaded and no swap
      if (!load_low || has_swap)
        store_features_values(start_index_low, feature_index, SIZE_BATCH,
                              feature_values_commit_low);
      if (!load_high || has_swap)
        store_features_values(start_index_high, feature_index, SIZE_BATCH,
                              feature_values_commit_high);

      start_index_low += SIZE_BATCH;
      assert(start_index_high > SIZE_BATCH);
      start_index_high -= SIZE_BATCH;
      if (start_index_high < start_index_low) {
        commit_loop = false;
        pivot = start_index_high;
#ifdef DEBUG
        printf("End no buffer left\n");
#endif
      } else if (start_index_low == start_index_high) {

        // only one buffer left
        // partition it
        feature_values_commit_low = load_feature_values(
            start_index_low, feature_index, SIZE_BATCH, feature_values[me()]);
        feature_values_commit_cmp_low =
            load_feature_values(start_index_low, cmp_feature_index, SIZE_BATCH,
                                feature_values2[me()]);
#ifdef DEBUG
        printf("Both case, one buffer left\n");
#endif
        pivot = partition_buffer(feature_values_commit_low,
                                 feature_values_commit_cmp_low, SIZE_BATCH,
                                 cmds_array[index_cmd].feature_threshold) +
                start_index_low;
        store_features_values(start_index_low, feature_index, SIZE_BATCH,
                              feature_values_commit_low);
        commit_loop = false;
      } else {
        load_low = true;
        load_high = true;
      }
    } else if (status == LOW_SWAP_COMPLETED) {

      // write low buffer back and get next one
      // if no next buffer, this is the end
      // reorder the high buffer correctly and write it back
      if (!load_low || has_swap)
        store_features_values(start_index_low, feature_index, SIZE_BATCH,
                              feature_values_commit_low);

      start_index_low += SIZE_BATCH;
      if (start_index_low >= start_index_high) {

        // no new buffer
        // paritition the high buffer and write it
        pivot = partition_buffer(feature_values_commit_high,
                                 feature_values_commit_cmp_high, SIZE_BATCH,
                                 cmds_array[index_cmd].feature_threshold) +
                start_index_high;
        store_features_values(start_index_high, feature_index, SIZE_BATCH,
                              feature_values_commit_high);
        commit_loop = false;
      } else {
        load_low = true;
        load_high = false;
      }
    } else if (status == HIGH_SWAP_COMPLETED) {

      // write high buffer back and get next one
      // if no next buffer, this is the end
      // reorder the low buffer correctly and write it back
      if (!load_high || has_swap)
        store_features_values(start_index_high, feature_index, SIZE_BATCH,
                              feature_values_commit_high);

      assert(start_index_high > SIZE_BATCH);
      start_index_high -= SIZE_BATCH;
      if (start_index_low >= start_index_high) {

        // no new buffer
        // paritition the low buffer and write it
        pivot = partition_buffer(feature_values_commit_low,
                                 feature_values_commit_cmp_low, SIZE_BATCH,
                                 cmds_array[index_cmd].feature_threshold) +
                start_index_low;
        store_features_values(start_index_low, feature_index, SIZE_BATCH,
                              feature_values_commit_low);
        commit_loop = false;
      } else {
        load_low = false;
        load_high = true;
      }
    }
  }

  // Here we need to handle the prolog and epilog due to alignment constraints
  // prolog
  // TODO

  // epilog
  mutex_lock(commit_mutex);
  for (int i = start_index + size; i < leaf_end_index[leaf_index]; ++i) {
#ifdef DEBUG
    printf("epilog %u\n", i);
#endif
    feature_t a, a_cmp, b;
    uint32_t epilog_index = feature_index * n_points + i;
    uint32_t epilog_index_cmp = cmp_feature_index * n_points + i;
    memcpy(&a_cmp, &t_features[epilog_index_cmp], sizeof(feature_t));
    if ((a_cmp < cmds_array[index_cmd].feature_threshold) &&
        (epilog_index != pivot)) {
      // need to swap with the pivot
      memcpy(&a, &t_features[epilog_index], sizeof(feature_t));
      memcpy(&b, &t_features[pivot], sizeof(feature_t));
      swap(&a, &b);
      memcpy(&t_features[epilog_index], &a, sizeof(feature_t));
      memcpy(&t_features[pivot], &b, sizeof(feature_t));
      pivot++;
    }
  }
  mutex_unlock(commit_mutex);

  // update leaf indexes
  if (self) {
    // create a new leaf only if not all elements fall
    // the same side of the threshold
    if (pivot > 0 && pivot < leaf_end_index[leaf_index]) {
#ifdef DEBUG
      printf("pivot found %u\n", pivot);
#endif
      n_leaves++;
      assert(n_leaves < MAX_NB_LEAF);
      uint32_t index_tmp = leaf_end_index[leaf_index];
      leaf_end_index[leaf_index] = pivot;
      leaf_start_index[n_leaves - 1] = pivot;
      leaf_end_index[n_leaves - 1] = index_tmp;
    }
  }

#ifdef DEBUG
  printf("do_split_commit end tid %d index_cmd %u feature_index %u\n", me(),
         index_cmd, feature_index);
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
  if (me() == 0) {
    int k = 0;
    n_points = 200;
    n_features = 5;
    for (int j = 0; j < n_features; ++j) {
      for (int i = 0; i < n_points; ++i) {
        if (i < n_points * 0.3)
          t_features[k] = 10;
        else
          t_features[k] = 50;
        k++;
      }
    }
    n_classes = 4;
    for (int i = 0; i < n_points; ++i) {
      t_targets[i] = i & 3;
      printf("point %u target %u\n", i, t_targets[i]);
    }
    nb_cmds = 2;
    struct Command cmd1, cmd2, cmd3;
    cmd1.type = 0;
    cmd1.feature_index = 0;
    cmd1.feature_threshold = 30;
    cmd1.leaf_index = 0;

    cmd2.type = 0;
    cmd2.feature_index = 2;
    cmd2.feature_threshold = 60;
    cmd2.leaf_index = 1;

    cmd3.type = 1;
    cmd3.feature_index = 3;
    cmd3.feature_threshold = 30;
    cmd3.leaf_index = 0;

    cmds_array[0] = cmd2;
    cmds_array[1] = cmd3;

    n_leaves = 2;
    leaf_start_index[0] = 0;
    leaf_end_index[0] = n_points >> 1;
    leaf_start_index[1] = n_points >> 1;
    leaf_end_index[1] = n_points;

    for (int l = 0; l < n_leaves; ++l) {
      printf("leaf %u:\n", l);
      for (uint32_t f = 0; f < n_features; ++f) {
        for (int i = leaf_start_index[l]; i < leaf_end_index[l]; ++i) {
          printf("feature %u: %f\n", f, t_features[f * n_points + i]);
        }
      }
    }
  }
  /////////////////////

  // initialization
  if (me() == 0) {
    batch_cnt = 0;
    cmd_cnt = 0;
    memset(cmd_tasklet_cnt, 0, nb_cmds);
    // TODO should we initialize gini_cnt to zero at each new launch
    // or let the host manage it?
  }
  barrier_wait(&barrier);

  uint16_t index_cmd = 0, last_index_cmd = 0;
  uint32_t index_batch = 0;

  while (get_next_batch(&index_cmd, &index_batch)) {

    if (cmds_array[index_cmd].type == SPLIT_EVALUATE) {
      do_split_evaluate(index_cmd, index_batch);
    } else if (cmds_array[index_cmd].type == SPLIT_COMMIT) {

      do_split_commit(index_cmd, index_batch);

      bool last = false;
      mutex_lock(commit_mutex);
      if (++cmd_tasklet_cnt[index_cmd] == n_features - 1)
        last = true;
      mutex_unlock(commit_mutex);

      if (last)
        do_split_commit(index_cmd, cmds_array[index_cmd].feature_index);
    } else {
      // TODO error
    }

    last_index_cmd = index_cmd;
  }

  barrier_wait(&barrier);

  //////////////// test
  if (me() == 0) {
    /*__dma_aligned uint32_t tmp_gini_cnt[MAX_CLASSES * 4];*/
    /*mram_read(gini_cnt, tmp_gini_cnt, n_leaves * n_classes *
     * sizeof(uint32_t));*/
    for (int l = 0; l < n_leaves; ++l) {
      printf("leaf %d:\n", l);
      for (int i = 0; i < n_classes; ++i)
        printf("gini count class %u: %u/%u\n", i, gini_cnt[l * n_classes + i],
               leaf_end_index[l] - leaf_start_index[l]);

      printf("split feature value:\n");
      for (int i = leaf_start_index[l]; i < leaf_end_index[l]; ++i) {
        printf("feature %u: %f\n", cmds_array[1].feature_index,
               t_features[cmds_array[1].feature_index * n_points + i]);
      }
      if (l == 0 || l == 2) {
        printf("other features value:\n");
        for (uint32_t f = 0; f < n_features; ++f) {
          if (f == cmds_array[1].feature_index)
            continue;
          for (int i = leaf_start_index[l]; i < leaf_end_index[l]; ++i) {
            printf("feature %u: %f\n", f, t_features[f * n_points + i]);
          }
        }
      }
    }
  }
  /////////////////////

  return 0;
}

#endif
