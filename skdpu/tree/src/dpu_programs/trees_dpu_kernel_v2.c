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

#include <attributes.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <assert.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>

#include "../trees_common.h"

/*#define DEBUG*/

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
__mram_noinit feature_t t_targets[MAX_SAMPLES_DPU]; /**< Vector of targets. */

/**
 * @brief an array of commands to execute
 * Note: there is no check that the host does not send two commands for the same
 * leaf However this would be undefined behavior if for instance some tasklets
 * are in parallel doing a SPLIT_EVALUATE and SPLIT_COMMIT for the same leaf
 **/
__host struct Command cmds_array[MAX_NB_LEAF];
/**
 * @brief number of commands in the array
 **/
__host uint16_t nb_cmds;

/**
 * @brief array to store the scores or gini count for each leaf and classes.
 * The first gini low count is found at element leaf_index * 2 * n_classes
 * The first gini high count is found at element (leaf_index * 2 + 1) *
 *n_classes
 **/
__mram uint32_t gini_cnt[MAX_NB_LEAF * 2 * MAX_CLASSES];

/**
 * @brief array to store the min/max feature values for each leaf.
 * This is the returned values for a SPLIT_MINMAX command.
 **/
__mram feature_t min_max_feature[MAX_NB_LEAF * 2];
MUTEX_INIT(minmax_mutex);

/**
 * @brief the points in one tree leaf are stored consecutively in MRAM
 * This order is maitained after a split commit by reordering the t_features and
 * the t_targets arrays in MRAM leaf_start_index is the index of the first point
 * in the tree leaf leaf_end_index is the end index of the tree leaf, the start
 * index of the next leaf
 **/
__host uint32_t n_leaves;
__host uint32_t start_n_leaves;
__host uint32_t leaf_start_index[MAX_NB_LEAF];
__host uint32_t leaf_end_index[MAX_NB_LEAF];
MUTEX_INIT(n_leaves_mutex);

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
__dma_aligned feature_t feature_values2[NR_TASKLETS][SIZE_BATCH + 16];
__dma_aligned feature_t feature_values3[NR_TASKLETS][SIZE_BATCH + 16];
__dma_aligned feature_t feature_values4[NR_TASKLETS][SIZE_BATCH + 16];

/**
 * @ brief WRAM buffer for classes values, SIZE_BATCH values per tasklet
 **/
__dma_aligned feature_t classes_values[NR_TASKLETS][SIZE_BATCH + 16];

/**
 * @brief WRAM buffers and mutex for the gini count
 **/
uint32_t w_gini_cnt_low[NR_TASKLETS][MAX_CLASSES];
uint32_t w_gini_cnt_high[NR_TASKLETS][MAX_CLASSES];
__dma_aligned uint32_t c_gini_cnt[NR_TASKLETS][2 * MAX_CLASSES];
MUTEX_INIT(gini_mutex);

/**
 * @brief batch counter and mutex
 **/
uint32_t batch_cnt = 0;
uint32_t cmd_cnt = 0;
MUTEX_INIT(batch_mutex);

/**
 * @brief initialization array for the min_max_feature array in MRAM
 **/
__dma_aligned feature_t min_max_init[2] = {FLT_MAX, 0};

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
  if (cmds_array[cmd_cnt].type == SPLIT_EVALUATE ||
      cmds_array[cmd_cnt].type == SPLIT_MINMAX) {
    uint16_t leaf_index = cmds_array[cmd_cnt].leaf_index;
    *index_batch = leaf_start_index[leaf_index];
  } else {
    // commit case
    *index_batch = 0;
  }
  return true;
}

static bool get_next_batch_evaluate(uint16_t *index_cmd, uint32_t *index_batch,
                                    bool *first) {

  uint16_t leaf_index = cmds_array[cmd_cnt].leaf_index;
  *index_batch = batch_cnt * SIZE_BATCH + leaf_start_index[leaf_index];
  *first = (cmd_cnt == 0) && (batch_cnt == 0);
  if (*index_batch >= leaf_end_index[leaf_index]) {
    // finished processing this command, go to next
    *first = get_next_command(index_cmd, index_batch);
    return *first;
  }
  *index_cmd = cmd_cnt;
  ++batch_cnt;
  return true;
}

static bool get_next_batch_commit(uint16_t *index_cmd, uint32_t *index_batch,
                                  bool *first) {

  *first = (cmd_cnt == 0) && (batch_cnt == 0);

  // The trick here is, the feature that has the split needs to be updated last
  // Otherwise we loose the information how to order for the other features
  // for the commit, the batch count is the next feature to handle
  if (batch_cnt == cmds_array[cmd_cnt].feature_index)
    batch_cnt++;

  // Note: the batch_cnt ends when equals to n_features +1
  // The reason is that a batch count equal to n_features is interpreted
  // as the targets. This means the tasklets will handle all features
  // expect the split feature, the targets, then synchronize and one of
  // them handle the split feature.
  if (batch_cnt > n_features) {
    *first = get_next_command(index_cmd, index_batch);
    return *first;
  }

  *index_cmd = cmd_cnt;
  *index_batch = batch_cnt;
  batch_cnt++;
  return true;
}

static bool get_next_batch(uint16_t *index_cmd, uint32_t *index_batch) {

  mutex_lock(batch_mutex);
  bool res = false;
  bool first_batch = false;
  if (cmd_cnt >= nb_cmds)
    res = false;
  else if (cmds_array[cmd_cnt].type == SPLIT_EVALUATE ||
           cmds_array[cmd_cnt].type == SPLIT_MINMAX) {
    res = get_next_batch_evaluate(index_cmd, index_batch, &first_batch);
  } else {
    // commit
    // no next batch, go to next command
    res = get_next_batch_commit(index_cmd, index_batch, &first_batch);
  }
  if (res && first_batch) {
    if (cmds_array[*index_cmd].type == SPLIT_EVALUATE) {
      // first batch of this SPLIT_EVALUATE command
      // Need to initialize the gini_cnt for this leaf
      uint32_t start_index = cmds_array[*index_cmd].leaf_index * 2 * n_classes;
      memset(&gini_cnt[start_index], 0, 2 * n_classes * sizeof(uint32_t));
    } else if (cmds_array[*index_cmd].type == SPLIT_MINMAX) {
      // initialize min_max_feature for this leaf
      // TODO here assume that feature_t is a multiple of 4
      uint32_t start_index = cmds_array[*index_cmd].leaf_index * 2;
      mram_write(min_max_init, &min_max_feature[start_index],
                 2 * sizeof(feature_t));
    }
  }
  mutex_unlock(batch_mutex);
  return res;
}

#define ALIGN_8_LOW(x) (((x) >> 3) << 3)
#define ALIGN_8_HIGH(x) (((x + 7) >> 3) << 3)

/**
 * @brief load features from MRAM to WRAM for a batch, while taking care of
 * alignment constraints
 * Note: for code simplicity we consider as of now that the targets are stored
 * as a feature type, hence this function also handles the targets load
 **/
static feature_t *load_feature_values(uint32_t index_batch,
                                      uint16_t feature_index,
                                      uint16_t size_batch,
                                      feature_t *feature_values) {

  // Need to handle the constraint that MRAM read must be on a MRAM address
  // aligned on 8 bytes with a size also aligned on 8 bytes

  bool is_target = feature_index == n_features;
  if (is_target)
    feature_index = 0;

  uint32_t start_index = feature_index * n_points + index_batch;
  uint32_t start_index_8align =
      ALIGN_8_LOW(start_index * sizeof(feature_t)) / sizeof(feature_t);
  uint32_t diff = start_index - start_index_8align;
  uint32_t size_batch_8align =
      ALIGN_8_HIGH((size_batch + diff) * sizeof(feature_t)) / sizeof(feature_t);

  if (is_target)
    mram_read(&t_targets[start_index_8align], feature_values,
              size_batch_8align * sizeof(feature_t));
  else
    mram_read(&t_features[start_index_8align], feature_values,
              size_batch_8align * sizeof(feature_t));

  return feature_values + diff;
}

/**
 * @brief load classes values from MRAM to WRAM for a batch, while taking care
 *of alignment constraints
 **/
static feature_t *load_classes_values(uint32_t index_batch,
                                      uint16_t size_batch) {

  // Need to handle the constraint that MRAM read must be on a MRAM address
  // aligned on 8 bytes with a size also aligned on 8 bytes

  return load_feature_values(index_batch, n_features, size_batch,
                             classes_values[me()]);
}

/**
 * @brief store features from MRAM to WRAM for a batch, while taking care of
 * alignment constraints
 * Warning: the WRAM buffer must be already in sync with the re-alignment done
 *in this function. This works when loading the features with
 *load_feature_values function
 **/
static void store_feature_values(uint32_t index_batch, uint16_t feature_index,
                                 uint16_t size_batch,
                                 feature_t *feature_values) {

  // Need to handle the constraint that MRAM read must be on a MRAM address
  // aligned on 8 bytes with a size also aligned on 8 bytes

  bool is_target = feature_index == n_features;
  if (is_target)
    feature_index = 0;

  uint32_t start_index = feature_index * n_points + index_batch;
  uint32_t start_index_8align =
      ALIGN_8_LOW(start_index * sizeof(feature_t)) / sizeof(feature_t);
  uint32_t diff = start_index - start_index_8align;
  uint32_t size_batch_8align =
      ALIGN_8_HIGH((size_batch + diff) * sizeof(feature_t)) / sizeof(feature_t);

  if (is_target)
    mram_write(feature_values, &t_targets[start_index_8align],
               size_batch_8align * sizeof(feature_t));
  else
    mram_write(feature_values, &t_features[start_index_8align],
               size_batch_8align * sizeof(feature_t));
}

/**
 * @brief update the gini counts in MRAM for a given leaf
 **/
static void update_gini_cnt(uint16_t leaf_index, uint32_t *gini_cnt_low,
                            uint32_t *gini_cnt_high) {

  mutex_lock(gini_mutex);
  // load the current gini cnt
  // Need to take into account the MRAM alignement constraints
#ifdef DEBUG
  printf("update gini count %u leaf %u\n", me(), leaf_index);
#endif
  uint32_t start_index = leaf_index * 2 * n_classes;
  mram_read(&gini_cnt[start_index], c_gini_cnt[me()],
            2 * n_classes * sizeof(uint32_t));

  // update gini low count
  for (int i = 0; i < n_classes; ++i) {
#ifdef DEBUG
    printf("class %u current count low %u=%u  incr %u\n", i,
           c_gini_cnt[me()][i], gini_cnt[start_index + i], gini_cnt_low[i]);
#endif
    c_gini_cnt[me()][i] += gini_cnt_low[i];
  }

  // update gini high count
  for (int i = n_classes; i < 2 * n_classes; ++i) {
#ifdef DEBUG
    printf("class %u current count high %u=%u  incr %u\n", i - n_classes,
           c_gini_cnt[me()][i], gini_cnt[start_index + i],
           gini_cnt_high[i - n_classes]);
#endif
    c_gini_cnt[me()][i] += gini_cnt_high[i - n_classes];
  }
  // store the updated values
  mram_write(c_gini_cnt[me()], &gini_cnt[start_index],
             2 * n_classes * sizeof(uint32_t));
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
  feature_t *classes_val = load_classes_values(index_batch, size_batch);

  memset(w_gini_cnt_low[me()], 0, n_classes * sizeof(uint32_t));
  memset(w_gini_cnt_high[me()], 0, n_classes * sizeof(uint32_t));

  for (int i = 0; i < size_batch; ++i) {

    if (features_val[i] <= cmds_array[index_cmd].feature_threshold) {

      // increment the gini count for this class
#ifdef DEBUG
      printf("tid %u point %u class %f %f feature %f threshold %f index_batch "
             "%u size_batch %u\n",
             me(), index_batch + i, classes_val[i], t_targets[index_batch + i],
             features_val[i], cmds_array[index_cmd].feature_threshold,
             index_batch, size_batch);
#endif
      w_gini_cnt_low[me()][(int32_t)(classes_val[i])]++;
    } else
      w_gini_cnt_high[me()][(int32_t)(classes_val[i])]++;
  }

  // update the gini count in MRAM
  update_gini_cnt(cmds_array[index_cmd].leaf_index, w_gini_cnt_low[me()],
                  w_gini_cnt_high[me()]);

#ifdef DEBUG
  printf("do_split_evaluate end %d cmd %u batch %u\n", me(), index_cmd,
         index_batch);
#endif
}

/**
 * @brief handle a batch for the SPLIT_MINMAX command
 **/
static void do_split_minmax(uint16_t index_cmd, uint32_t index_batch) {

#ifdef DEBUG
  printf("do_split_minmax %d cmd %u batch %u\n", me(), index_cmd, index_batch);
#endif
  uint16_t size_batch = SIZE_BATCH;
  if (index_batch + SIZE_BATCH >
      leaf_end_index[cmds_array[index_cmd].leaf_index])
    size_batch = leaf_end_index[cmds_array[index_cmd].leaf_index] - index_batch;

  feature_t *features_val =
      load_feature_values(index_batch, cmds_array[index_cmd].feature_index,
                          size_batch, feature_values[me()]);

  float min = FLT_MAX, max = 0;
  for (int i = 0; i < size_batch; ++i) {

    if (features_val[i] < min) {
      // store as minimum value
      min = features_val[i];
    }
    if (features_val[i] > max) {
      // store as maximum value
      max = features_val[i];
    }
  }

  // update the min max in MRAM
  mutex_lock(minmax_mutex);
  __dma_aligned feature_t c_minmax[2];
  uint16_t leaf_index = cmds_array[index_cmd].leaf_index;
  // TODO here assume that sizeof(feature_t) is a multiple of 4 bytes
  mram_read(&min_max_feature[leaf_index * 2], c_minmax, 2 * sizeof(feature_t));
  bool write = false;
  if (min < c_minmax[0]) {
    c_minmax[0] = min;
    write = true;
  }
  if (max > c_minmax[1]) {
    c_minmax[1] = max;
    write = true;
  }
#ifdef DEBUG
  printf("do_split_minmax tid %d cmd %u batch %u: mram min %f batch min %f "
         "mram max %f batch max %f write %d\n",
         me(), index_cmd, index_batch, c_minmax[0], min, c_minmax[1], max,
         write);
#endif
  if (write)
    mram_write(c_minmax, &min_max_feature[leaf_index * 2],
               2 * sizeof(feature_t));
  mutex_unlock(minmax_mutex);

#ifdef DEBUG
  printf("do_split_minmax end %d cmd %u batch %u\n", me(), index_cmd,
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

/**
 * @brief swap the values of two buffers in order to obtain a buffer
 * with values lower than the threshold and values higher than the threshold.
 * Then function stops swapping when one of the buffer is full (i.e., contains
 * only values lower or higher).
 * The buffer used to compare to the threshold may be different that the buffer
 * on which we actually swap the values.
 **/
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

/**
 * @brief function that determines the start index and the size on which
 * the swap algorithm will work. The rest will be a prolog and epilog which
 * are handled separatly. This is needed due to the alignment constraints in
 * MRAM/WRAM transferts.
 **/
static void get_index_and_size_for_commit(uint16_t index_cmd,
                                          uint16_t feature_index,
                                          uint32_t *index, uint32_t *size) {

  uint16_t leaf_index = cmds_array[index_cmd].leaf_index;
  uint32_t start_index =
      feature_index * n_points + leaf_start_index[leaf_index];
  uint32_t end_index = leaf_end_index[leaf_index];
  *index = (ALIGN_8_HIGH(start_index * sizeof(feature_t)) / sizeof(feature_t)) -
           (feature_index * n_points);
  // handle the case where *index > end_index
  if (*index >= end_index) {
    *size = end_index - start_index;
    return;
  }
  uint32_t nb_buffers = (end_index - *index) / SIZE_BATCH;
  if (nb_buffers)
    *size = nb_buffers * SIZE_BATCH;
  else
    *size = end_index - start_index;
}

/**
 * @brief write back some feature values to MRAM. Does not
 * enforce the size and address 8-bytes alignment.
 **/
static void store_feature_values_noalign(uint32_t start_index,
                                         uint32_t feature_index,
                                         uint32_t size_batch,
                                         feature_t *feature_values) {

  bool is_target = feature_index == n_features;

  // feature_index * n_points * sizeof(feature_t) must be aligned on 8
  // bytes. Currently this is enforced through an assert in main function
  if (!is_target)
    assert(((uint32_t)(&t_features[feature_index * n_points + start_index]) &
            7) == 0);
  assert((size_batch * sizeof(feature_t) & 7) == 0);

  if (is_target)
    mram_write(feature_values, &t_targets[start_index],
               size_batch * sizeof(feature_t));
  else
    mram_write(feature_values,
               &t_features[feature_index * n_points + start_index],
               size_batch * sizeof(feature_t));
}

/**
 * @brief partition a buffer of feature values in values lower than the
 * threshold (to the left) and values higher than the threshold (to the right).
 * The buffer used for comparison to the threshold may be different than the
 * buffer on which values are reordered.
 **/
static int partition_buffer(feature_t *feature_values,
                            feature_t *feature_values_cmp, uint32_t size_batch,
                            feature_t feature_threshold) {

  int pivot = -1;
  assert(size_batch);
  for (int j = 0; j < size_batch; j++) {
    if (feature_values_cmp[j] <= feature_threshold) {
      pivot++;
      if (pivot != j) {
        swap(feature_values + pivot, feature_values + j);
      }
    }
  }
  return ++pivot;
}

/**
 * @brief handle a batch for the split commit command
 * feature values in MRAM are reordered to keep values of the same leaf
 * consecutive. One call of this function is handling the reordering for
 * one feature.
 * The feature used for the split is handled last
 * as the other features are reordered based on these values.
 **/
static void do_split_commit(uint16_t index_cmd, uint32_t feature_index,
                            uint16_t index_new_leaf) {

#ifdef DEBUG
  printf("do_split_commit tid %d index_cmd %u feature_index %u\n", me(),
         index_cmd, feature_index);
#endif

  uint32_t start_index = 0, size = 0;
  bool is_target = feature_index == n_features;

  // the main algorithm will work on aligned indexes, this means
  // the first index is aligned on 8 bytes, and the size is aligned on
  // SIZE_BATCH In the end we handle the prolog and epilog
  get_index_and_size_for_commit(index_cmd, feature_index, &start_index, &size);
  uint16_t leaf_index = cmds_array[index_cmd].leaf_index;
  uint16_t cmp_feature_index = cmds_array[index_cmd].feature_index;
  bool self = feature_index == cmp_feature_index;

  bool commit_loop = true;
  uint32_t start_index_low = start_index;
  uint32_t start_index_high =
      (size >= SIZE_BATCH) ? start_index + size - SIZE_BATCH : start_index;
  feature_t *feature_values_commit_low, *feature_values_commit_cmp_low;
  feature_t *feature_values_commit_high, *feature_values_commit_cmp_high;
  int pivot;

  // only one buffer, directly partition it
  // It can be the size of a batch (in which case it is aligned)
  // Or it can be less in which case it is not aligned
  if (start_index_low == start_index_high) {

    assert(size <= SIZE_BATCH);

    // if not aligned take mutex
    if (size < SIZE_BATCH)
      mutex_lock(commit_mutex);

    feature_values_commit_low = load_feature_values(
        start_index_low, feature_index, size, feature_values[me()]);
    if (!self)
      feature_values_commit_cmp_low = load_feature_values(
          start_index_low, cmp_feature_index, size, feature_values2[me()]);
    else
      feature_values_commit_cmp_low = feature_values_commit_low;

    pivot = partition_buffer(feature_values_commit_low,
                             feature_values_commit_cmp_low, size,
                             cmds_array[index_cmd].feature_threshold) +
            start_index_low;

    store_feature_values(start_index_low, feature_index, size,
                         feature_values[me()]);

    if (size < SIZE_BATCH)
      mutex_unlock(commit_mutex);

    commit_loop = false;
  }

  bool load_low = true, load_high = true;
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
        store_feature_values_noalign(start_index_low, feature_index, SIZE_BATCH,
                                     feature_values_commit_low);
      if (!load_high || has_swap)
        store_feature_values_noalign(start_index_high, feature_index,
                                     SIZE_BATCH, feature_values_commit_high);

      start_index_low += SIZE_BATCH;
      assert(start_index_high >= SIZE_BATCH);
      start_index_high -= SIZE_BATCH;
      if (start_index_high < start_index_low) {
        commit_loop = false;
        pivot = start_index_low;
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
        store_feature_values_noalign(start_index_low, feature_index, SIZE_BATCH,
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
        store_feature_values_noalign(start_index_low, feature_index, SIZE_BATCH,
                                     feature_values_commit_low);

      start_index_low += SIZE_BATCH;
      if (start_index_low >= start_index_high) {

        // no new buffer
        // paritition the high buffer and write it
        pivot = partition_buffer(feature_values_commit_high,
                                 feature_values_commit_cmp_high, SIZE_BATCH,
                                 cmds_array[index_cmd].feature_threshold) +
                start_index_high;
        store_feature_values_noalign(start_index_high, feature_index,
                                     SIZE_BATCH, feature_values_commit_high);
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
        store_feature_values_noalign(start_index_high, feature_index,
                                     SIZE_BATCH, feature_values_commit_high);

      assert(start_index_high >= SIZE_BATCH);
      start_index_high -= SIZE_BATCH;
      if (start_index_low >= start_index_high) {

        // no new buffer
        // paritition the low buffer and write it
        pivot = partition_buffer(feature_values_commit_low,
                                 feature_values_commit_cmp_low, SIZE_BATCH,
                                 cmds_array[index_cmd].feature_threshold) +
                start_index_low;
        store_feature_values_noalign(start_index_low, feature_index, SIZE_BATCH,
                                     feature_values_commit_low);
        commit_loop = false;
      } else {
        load_low = false;
        load_high = true;
      }
    }
  }

  // Here we need to handle the prolog and epilog due to alignment constraints
  mutex_lock(commit_mutex);
  // prolog
  uint32_t prolog_start = leaf_start_index[leaf_index];
  uint32_t prolog_end = start_index;
  if (prolog_start < prolog_end) {
#ifdef DEBUG
    printf("prolog\n");
#endif
    feature_t *feature_values_prolog =
        load_feature_values(prolog_start, feature_index,
                            prolog_end - prolog_start, feature_values[me()]);
    feature_t *feature_values_cmp =
        load_feature_values(prolog_start, cmp_feature_index,
                            prolog_end - prolog_start, feature_values2[me()]);

    uint32_t start_swap_buffer;
    uint32_t max_n_elems_pivot = prolog_end - prolog_start;
    if (pivot - start_index < max_n_elems_pivot) {
      // not enough space after the pivot
      max_n_elems_pivot = pivot - start_index;
      start_swap_buffer = start_index;
    } else
      start_swap_buffer = pivot - (prolog_end - prolog_start);

    feature_t *feature_values_pivot =
        load_feature_values(start_swap_buffer, feature_index, max_n_elems_pivot,
                            feature_values3[me()]);

    uint32_t swap_index = 0;
    for (int i = 0; i < prolog_end - prolog_start; ++i) {
      if (feature_values_cmp[i] > cmds_array[index_cmd].feature_threshold) {
        assert(pivot);
        swap(&feature_values_prolog[i],
             &feature_values_pivot[max_n_elems_pivot - (++swap_index)]);
        if (swap_index >= max_n_elems_pivot)
          break;
      }
    }

    assert(pivot >= swap_index);
    pivot -= swap_index;
    if (pivot == start_index) {
      // need to also partition the remainder of the prolog
      uint32_t pivot_prolog = partition_buffer(
          feature_values_prolog, feature_values_cmp, prolog_end - prolog_start,
          cmds_array[index_cmd].feature_threshold);
      pivot = start_index + pivot_prolog;
      max_n_elems_pivot = prolog_end - prolog_start;
    }

    // Need to store the values back in MRAM, but need to handle
    // misalignement. We need to make sure no other tasklet is writting at the
    // same time. But if the misaligned part that we may erase (not part of the
    // same leaf) is in the epilog of the contiguous leaf, and since the
    // prolog/epilog handling is in a critical section protected by a mutex, we
    // avoid the issue. Making sure that the misaligned part is in the epilog of
    // the other leaf requires that SIZE_BATCH * sizeof(feature_t) > 8 => this
    // is asserted in the main function
    store_feature_values(prolog_start, feature_index, prolog_end - prolog_start,
                         feature_values[me()]);

    store_feature_values(start_swap_buffer, feature_index, max_n_elems_pivot,
                         feature_values3[me()]);
  }
  // epilog
  uint32_t epilog_start = start_index + size;
  uint32_t epilog_end = leaf_end_index[leaf_index];
  if (epilog_start < epilog_end) {
#ifdef DEBUG
    printf("epilog\n");
#endif
    feature_t *feature_values_epilog =
        load_feature_values(epilog_start, feature_index,
                            epilog_end - epilog_start, feature_values[me()]);
    feature_t *feature_values_cmp =
        load_feature_values(epilog_start, cmp_feature_index,
                            epilog_end - epilog_start, feature_values2[me()]);

    uint32_t max_n_elems_pivot = epilog_end - epilog_start;
    if (pivot + max_n_elems_pivot > epilog_start) {
      // not enough space after the pivot
      assert(pivot <= epilog_start);
      max_n_elems_pivot = epilog_start - pivot;
    }
    feature_t *feature_values_pivot = load_feature_values(
        pivot, feature_index, max_n_elems_pivot, feature_values3[me()]);

    uint32_t swap_index = 0;
    for (int i = 0; i < epilog_end - epilog_start; ++i) {
      if (feature_values_cmp[i] <= cmds_array[index_cmd].feature_threshold) {
        swap(&feature_values_epilog[i], &feature_values_pivot[swap_index++]);
        if (swap_index >= max_n_elems_pivot)
          break;
      }
    }

    uint32_t old_pivot = pivot;
    pivot += swap_index;
    if (pivot >= epilog_start) {
      // need to also partition the remainder of the epilog
      pivot += partition_buffer(feature_values_epilog, feature_values_cmp,
                                epilog_end - epilog_start,
                                cmds_array[index_cmd].feature_threshold);
      max_n_elems_pivot = epilog_end - epilog_start;
    }
    store_feature_values(epilog_start, feature_index, epilog_end - epilog_start,
                         feature_values[me()]);

    store_feature_values(old_pivot, feature_index, max_n_elems_pivot,
                         feature_values3[me()]);
  }
  mutex_unlock(commit_mutex);

  // update leaf indexes
  if (self) {
    // create a new leaf even if not all elements fall
    // the same side of the threshold
    // In this case it will be empty
#ifdef DEBUG
    printf("pivot found %u\n", pivot);
#endif
    assert(index_new_leaf < MAX_NB_LEAF);
    uint32_t index_tmp = leaf_end_index[leaf_index];
    leaf_end_index[leaf_index] = pivot;
    leaf_start_index[index_new_leaf] = pivot;
    leaf_end_index[index_new_leaf] = index_tmp;
    mutex_lock(n_leaves_mutex);
    n_leaves++;
    mutex_unlock(n_leaves_mutex);
  }

#ifdef DEBUG
  printf("do_split_commit end tid %d index_cmd %u feature_index %u\n", me(),
         index_cmd, feature_index);
#endif
}

/**
 * @return the next leaf index to use for a commit command
 **/
static uint16_t get_new_leaf_index(uint16_t index_cmd) {

  uint16_t n_commits = 0;
  for (int i = 0; i < index_cmd; ++i) {
    if (cmds_array[i].type == SPLIT_COMMIT)
      n_commits++;
  }
  return start_n_leaves + n_commits;
}

BARRIER_INIT(barrier, NR_TASKLETS);

// define the TEST variable to run the unit tests
#ifdef TEST1
#include "./test/test1.h"
#elif defined(TEST2)
#include "./test/test2.h"
#elif defined(TEST3)
#include "./test/test3.h"
#elif defined(TEST4)
#include "./test/test4.h"
#elif defined(TEST5)
#include "./test/test5.h"
#elif defined(TEST6)
#include "./test/test6.h"
#elif defined(TEST7)
#include "./test/test7.h"
#endif

/*================== MAIN FUNCTION ======================*/
/**
 * @brief Main function DPU side.
 *
 * @return int 0 on success.
 */
int main() {

#ifdef TEST
  if (me() == 0) {
    test_init();
  }
#endif

  _Static_assert((SIZE_BATCH & 7) == 0, "SIZE_BATCH must be a multiple of 8");
  _Static_assert(SIZE_BATCH * sizeof(feature_t) > 8,
                 "Please make sure to satisfy this condition necessary for "
                 "the commit command");
  _Static_assert((sizeof(feature_t) & 3) == 0,
                 "sizeof(feature_t) must be multiple of 4");

  // initialization
  if (me() == 0) {
    batch_cnt = 0;
    cmd_cnt = 0;
    memset(cmd_tasklet_cnt, 0, nb_cmds);
    start_n_leaves = n_leaves;
  }
  barrier_wait(&barrier);

  uint16_t index_cmd = 0;
  uint32_t index_batch = 0;

  while (get_next_batch(&index_cmd, &index_batch)) {

    if (cmds_array[index_cmd].type == SPLIT_EVALUATE) {

      do_split_evaluate(index_cmd, index_batch);

    } else if (cmds_array[index_cmd].type == SPLIT_COMMIT) {

      // if the targeted leaf is empty, this is an error
      uint16_t leaf_index = cmds_array[cmd_cnt].leaf_index;
      assert(leaf_start_index[leaf_index] < leaf_end_index[leaf_index]);

      do_split_commit(index_cmd, index_batch, 0);

      bool last = false;
      mutex_lock(commit_mutex);
      if (++cmd_tasklet_cnt[index_cmd] == n_features)
        last = true;
      mutex_unlock(commit_mutex);

      if (last)
        do_split_commit(index_cmd, cmds_array[index_cmd].feature_index,
                        get_new_leaf_index(index_cmd));

    } else if (cmds_array[index_cmd].type == SPLIT_MINMAX) {

      do_split_minmax(index_cmd, index_batch);

    } else {
      // TODO error handling
      assert(0);
    }
  }

  barrier_wait(&barrier);

#ifdef TEST
  if (me() == 0) {
    test_check();
  }
  barrier_wait(&barrier);
#endif

  return 0;
}

#endif
