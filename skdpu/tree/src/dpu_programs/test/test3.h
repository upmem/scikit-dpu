/**
 * @file test3.h
 * @author Julien Legriel (jlegriel@upmem.com)
 * @brief Test 3 for the DPU side of the tree algorithm
 *
 */

#ifndef _TREES_DPU_KERNEL_TEST3_H_
#define _TREES_DPU_KERNEL_TEST3_H_

#include "./test.h"

static void test_init() {

  printf("--------------- TEST3 ----------------------\n");

  // set points/classes
  if (me() == 0) {
    int k = 0;
    n_points = 200;
    n_features = 5;
    for (int j = 0; j < n_features; ++j) {
      for (int i = 0; i < n_points; ++i) {
        if ((i < n_points * 0.6) && (i & 1))
          t_features[k] = 10;
        else
          t_features[k] = 50;
        k++;
      }
    }
    n_classes = 4;
    for (int i = 0; i < n_points; ++i) {
      t_targets[i] = i & 3;
      printf("point %u target %f\n", i, t_targets[i]);
    }
    nb_cmds = 2;
    struct Command cmd2, cmd3;

    cmd2.type = 0;
    cmd2.feature_index = 2;
    cmd2.feature_threshold = 30;
    cmd2.leaf_index = 1;

    cmd3.type = 1;
    cmd3.feature_index = 3;
    cmd3.feature_threshold = 30;
    cmd3.leaf_index = 0;

    cmds_array[0] = cmd2;
    cmds_array[1] = cmd3;

    n_leaves = 2;
    leaf_start_index[0] = 0;
    // the tree leaf end is going to be disaligned
    leaf_end_index[0] = (n_points >> 1) - 1;
    leaf_start_index[1] = (n_points >> 1) - 1;
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
}
static void test_check() {

  assert(n_leaves == 3);
  uint32_t split_feature = 3;
  uint32_t *expected_gini_cnt =
      mem_alloc(n_leaves * n_classes * sizeof(uint32_t));
  uint32_t *expected_leaf_cnt = mem_alloc(n_leaves * sizeof(uint32_t));
  float *expected_feature_values =
      mem_alloc(n_features * n_points * sizeof(float));
  for (int i = 0; i < n_classes; ++i) {
    expected_gini_cnt[i] = 0;
    expected_gini_cnt[2 * n_classes + i] = 0;
  }
  expected_gini_cnt[n_classes + 0] = 0;
  expected_gini_cnt[n_classes + 1] = 5;
  expected_gini_cnt[n_classes + 2] = 0;
  expected_gini_cnt[n_classes + 3] = 6;

  expected_leaf_cnt[0] = 49;
  expected_leaf_cnt[1] = 101;
  expected_leaf_cnt[2] = 50;

  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_points; ++j) {
      if (j < 49)
        expected_feature_values[i * n_points + j] = 10;
      else if (j >= 49 && j < 99)
        expected_feature_values[i * n_points + j] = 50;
      else {
        if (j >= 120 || (j & 1) == 0)
          expected_feature_values[i * n_points + j] = 50;
        else
          expected_feature_values[i * n_points + j] = 10;
      }
    }
  }

  test_check_func(expected_gini_cnt, expected_leaf_cnt, split_feature,
                  expected_feature_values, 0);

  mem_reset();
}


#endif


