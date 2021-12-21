/**
 * @file test7.h
 * @author Julien Legriel (jlegriel@upmem.com)
 * @brief Test 7 for the DPU side of the tree algorithm
 *
 */

#ifndef _TREES_DPU_KERNEL_TEST7_H_
#define _TREES_DPU_KERNEL_TEST7_H_

#include "./test.h"

static void test_init() {

  printf("--------------- TEST7 ----------------------\n");

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
      printf("point %u target %f\n", i, t_targets[i]);
    }
    nb_cmds = 2;
    struct Command cmd2, cmd3;

    // test the SPLIT_MINMAX command
    cmd2.type = 2;
    cmd2.feature_index = 2;
    cmd2.feature_threshold = 60;
    cmd2.leaf_index = 1;

    cmd3.type = 2;
    cmd3.feature_index = 3;
    cmd3.feature_threshold = 30;
    cmd3.leaf_index = 0;

    cmds_array[0] = cmd3;
    cmds_array[1] = cmd2;

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
}

static void test_check() {

  assert(n_leaves == 2);
  uint32_t split_feature = 3;
  float *expected_min_max = mem_alloc(n_leaves * 2 * sizeof(float));

  // leaf 0
  expected_min_max[0 * 2] = 10;
  expected_min_max[0 * 2 + 1] = 50;

  // leaf 1
  expected_min_max[1 * 2] = 50;
  expected_min_max[1 * 2 + 1] = 50;

  test_check_func(0, 0, split_feature, 0, 0, expected_min_max);

  mem_reset();
}

#endif
