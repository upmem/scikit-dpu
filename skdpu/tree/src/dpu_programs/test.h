/**
 * @file trees_dpu_kernel_v2.c
 * @author Julien Legriel (jlegriel@upmem.com)
 * @brief Test for the DPU side of the tree algorithm
 *
 */

#ifndef _TREES_DPU_KERNEL_TEST_H_
#define _TREES_DPU_KERNEL_TEST_H_

#include <alloc.h>

// use this define to get print in case of failure instead of an assert
//#define PRINT_ERROR_NO_ASSERT

static void test_check_func(uint32_t *expected_gini_cnt_low,
                            uint32_t *expected_leaf_cnt, uint16_t split_feature,
                            float *expected_feature_values) {

  if (me() == 0) {
    for (int l = 0; l < start_n_leaves; ++l) {
      // only check the gini count of leaf that have SPLIT_EVALUATE command
      if (cmds_array[l].type != SPLIT_EVALUATE)
        continue;
      printf("leaf %d:\n", l);
      for (int i = 0; i < n_classes; ++i) {
        printf("gini count class %u: %u/%u\n", i,
               gini_cnt[l * 2 * n_classes + i],
               leaf_end_index[l] - leaf_start_index[l]);
#ifdef PRINT_ERROR_NO_ASSERT
        if (gini_cnt[l * 2 * n_classes + i] !=
            expected_gini_cnt_low[l * n_classes + i])
          printf("ASSERT: gini cnt leaf %u class %u found %u expected %u\n", l,
                 i, gini_cnt[l * 2 * n_classes + i],
                 expected_gini_cnt_low[l * n_classes + i]);
        if ((leaf_end_index[l] - leaf_start_index[l]) != expected_leaf_cnt[l])
          printf("ASSERT: leaf %u size found %u expected %u\n", l,
                 (leaf_end_index[l] - leaf_start_index[l]),
                 expected_leaf_cnt[l]);
#else
        assert(gini_cnt[l * 2 * n_classes + i] ==
               expected_gini_cnt_low[l * n_classes + i]);
        assert((leaf_end_index[l] - leaf_start_index[l]) ==
               expected_leaf_cnt[l]);
#endif
      }

      printf("split feature value:\n");
      for (int i = leaf_start_index[l]; i < leaf_end_index[l]; ++i) {
        printf("feature %u: %f\n", split_feature,
               t_features[split_feature * n_points + i]);
#ifdef PRINT_ERROR_NO_ASSERT
        if (t_features[split_feature * n_points + i] !=
            expected_feature_values[split_feature * n_points + i])
          printf("ASSERT: feature value at index %u found %f expected %f\n", i,
                 t_features[split_feature * n_points + i],
                 expected_feature_values[split_feature * n_points + i]);
#else
        assert(t_features[split_feature * n_points + i] ==
               expected_feature_values[split_feature * n_points + i]);
#endif
      }
      printf("other features value:\n");
      for (uint32_t f = 0; f < n_features; ++f) {
        if (f == split_feature)
          continue;
        for (int i = leaf_start_index[l]; i < leaf_end_index[l]; ++i) {
          printf("feature %u: %f\n", f, t_features[f * n_points + i]);
#ifdef PRINT_ERROR_NO_ASSERT
          if (t_features[split_feature * n_points + i] !=
              expected_feature_values[split_feature * n_points + i])
            printf("ASSERT: feature value at index %u found %f expected %f\n",
                   i, t_features[f * n_points + i],
                   expected_feature_values[f * n_points + i]);
#else
          assert(t_features[f * n_points + i] ==
                 expected_feature_values[f * n_points + i]);
#endif
        }
      }
    }
  }
  printf("--------------- TEST END -------------------\n");
}

static void test1_init() {

  printf("--------------- TEST1 ----------------------\n");

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
}

static void test1_check() {

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
  for (int i = 0; i < n_classes; ++i)
    expected_gini_cnt[1 * n_classes + i] = 25;

  expected_leaf_cnt[0] = 60;
  expected_leaf_cnt[1] = 100;
  expected_leaf_cnt[2] = 40;

  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_points; ++j) {
      if (j < 60)
        expected_feature_values[i * n_points + j] = 10;
      else
        expected_feature_values[i * n_points + j] = 50;
    }
  }

  test_check_func(expected_gini_cnt, expected_leaf_cnt, split_feature,
                  expected_feature_values);

  mem_reset();
}

static void test2_init() {

  printf("--------------- TEST2 ----------------------\n");

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

static void test2_check() {

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
  for (int i = 0; i < n_classes; ++i)
    expected_gini_cnt[1 * n_classes + i] = (i & 1) ? 5 : 0;

  expected_leaf_cnt[0] = 50;
  expected_leaf_cnt[1] = 100;
  expected_leaf_cnt[2] = 50;

  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_points; ++j) {
      if (j < 50)
        expected_feature_values[i * n_points + j] = 10;
      else if (j >= 50 && j < 100)
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
                  expected_feature_values);

  mem_reset();
}

static void test3_init() {

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
static void test3_check() {

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
                  expected_feature_values);

  mem_reset();
}

static void test4_init() {

  printf("--------------- TEST4 ----------------------\n");

  // set points/classes
  if (me() == 0) {
    int k = 0;
    n_points = 200;
    n_features = 5;
    for (int j = 0; j < n_features; ++j) {
      for (int i = 0; i < n_points; ++i) {
        if ((i > n_points * 0.6) && (i & 1))
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
    cmd2.leaf_index = 0;

    cmd3.type = 1;
    cmd3.feature_index = 3;
    cmd3.feature_threshold = 30;
    cmd3.leaf_index = 1;

    cmds_array[0] = cmd2;
    cmds_array[1] = cmd3;

    n_leaves = 2;
    leaf_start_index[0] = 0;
    // the tree leaf start of the split commit is going to be disaligned
    // This forces the prolog case
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

static void test4_check() {

  assert(n_leaves == 3);
  uint32_t split_feature = 3;
  uint32_t *expected_gini_cnt =
      mem_alloc(n_leaves * n_classes * sizeof(uint32_t));
  uint32_t *expected_leaf_cnt = mem_alloc(n_leaves * sizeof(uint32_t));
  float *expected_feature_values =
      mem_alloc(n_features * n_points * sizeof(float));
  for (int i = 0; i < n_classes; ++i) {
    expected_gini_cnt[i] = 0;
    expected_gini_cnt[n_classes + i] = 0;
    expected_gini_cnt[2 * n_classes + i] = 0;
  }

  expected_leaf_cnt[0] = 99;
  expected_leaf_cnt[1] = 40;
  expected_leaf_cnt[2] = 61;

  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_points; ++j) {
      if (j >= 99 && j < 139)
        expected_feature_values[i * n_points + j] = 10;
      else {
        expected_feature_values[i * n_points + j] = 50;
      }
    }
  }

  test_check_func(expected_gini_cnt, expected_leaf_cnt, split_feature,
                  expected_feature_values);

  mem_reset();
}

static void test5_init() {

  printf("--------------- TEST5 ----------------------\n");

  // set points/classes
  if (me() == 0) {
    int k = 0;
    n_points = 200;
    n_features = 5;
    for (int j = 0; j < n_features; ++j) {
      for (int i = 0; i < n_points; ++i) {
        if (i < 30 || (i >= 100 && i < 130))
          t_features[k] = (j & 1) ? ((i != 5 && i != 9) ? 10 : 50) : 50;
        else
          t_features[k] = (j & 1) ? 50 : 10;
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
    cmd2.leaf_index = 0;

    cmd3.type = 1;
    cmd3.feature_index = 3;
    cmd3.feature_threshold = 30;
    cmd3.leaf_index = 1;

    cmds_array[0] = cmd2;
    cmds_array[1] = cmd3;

    n_leaves = 2;
    leaf_start_index[0] = 0;
    // the tree leaf start of the split commit is going to be disaligned
    // This forces the prolog case
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

static void test5_check() {

  assert(n_leaves == 3);
  uint32_t split_feature = 3;
  uint32_t *expected_gini_cnt =
      mem_alloc(n_leaves * n_classes * sizeof(uint32_t));
  uint32_t *expected_leaf_cnt = mem_alloc(n_leaves * sizeof(uint32_t));
  float *expected_feature_values =
      mem_alloc(n_features * n_points * sizeof(float));
  expected_gini_cnt[0] = 17;
  expected_gini_cnt[1] = 17;
  expected_gini_cnt[2] = 18;
  expected_gini_cnt[3] = 17;
  for (int i = 0; i < n_classes; ++i) {
    expected_gini_cnt[n_classes + i] = 0;
    expected_gini_cnt[2 * n_classes + i] = 0;
  }

  expected_leaf_cnt[0] = 99;
  expected_leaf_cnt[1] = 30;
  expected_leaf_cnt[2] = 71;

  for (int j = 0; j < n_features; ++j) {
    for (int i = 0; i < n_points; ++i) {
      if (i < 99) { // leaf 0
        if (i < 30 || (i >= 100 && i < 130))
          expected_feature_values[j * n_points + i] =
              (j & 1) ? ((i != 5 && i != 9) ? 10 : 50) : 50;
        else
          expected_feature_values[j * n_points + i] = (j & 1) ? 50 : 10;
      } else if (i >= 99 && i < 129) { // leaf 1
        expected_feature_values[j * n_points + i] = (j & 1) ? 10 : 50;
      } else { // leaf 2
        expected_feature_values[j * n_points + i] = (j & 1) ? 50 : 10;
      }
    }
  }

  test_check_func(expected_gini_cnt, expected_leaf_cnt, split_feature,
                  expected_feature_values);

  mem_reset();
}

#define MAX_NB_TEST 20

/**
 * number of tests
 **/
uint32_t n_tests = 5;
uint32_t start_index_tests = 0;

/**
 * init functions
 **/
void (*test_init[])(void) = {test1_init, test2_init, test3_init, test4_init,
                             test5_init};

/**
 * check functions
 **/
void (*test_check[])(void) = {test1_check, test2_check, test3_check,
                              test4_check, test5_check};

#endif
