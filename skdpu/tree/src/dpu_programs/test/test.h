/**
 * @file test.h
 * @author Julien Legriel (jlegriel@upmem.com)
 * @brief Test utility function for the DPU side of the tree algorithm
 *
 */

#ifndef _TREES_DPU_KERNEL_TEST_H_
#define _TREES_DPU_KERNEL_TEST_H_

#include <alloc.h>

// use this define to get print in case of failure instead of an assert
//#define PRINT_ERROR_NO_ASSERT

static void test_check_func(uint32_t *expected_gini_cnt_low,
                            uint32_t *expected_leaf_cnt, uint16_t split_feature,
                            float *expected_feature_values,
                            float *expected_target_values) {

  if (me() == 0) {
    for (int l = 0; l < n_leaves; ++l) {

      printf("leaf %d:\n", l);

      // only check the gini count of leaf that have SPLIT_EVALUATE command
      if (l < start_n_leaves && cmds_array[l].type == SPLIT_EVALUATE) {
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
      }

      if (l >= start_n_leaves || cmds_array[l].type == SPLIT_COMMIT) {
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
        if(expected_target_values) {
          printf("targets value:\n");
          for (int i = leaf_start_index[l]; i < leaf_end_index[l]; ++i) {
            printf("target: %f\n", t_targets[i]);
#ifdef PRINT_ERROR_NO_ASSERT
            if (t_targets[i] !=
                expected_target_values[i])
              printf("ASSERT: class value at index %u found %f expected %f\n",
                  i, t_targets[i],
                  expected_target_values[i]);
#else
            assert(t_targets[i] ==
                expected_target_values[i]);
#endif
          }
        }
      }
    }
  }
  printf("--------------- TEST END -------------------\n");
}

#endif
