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
                            float *expected_target_values,
                            float *expected_min_max) {

  if (me() == 0) {
    for (int l = 0; l < n_leaves; ++l) {

      printf("leaf %d:\n", l);

      // only check the gini count of leaf that have SPLIT_EVALUATE command
      if (l < start_n_leaves && cmds_array[l].type == SPLIT_EVALUATE &&
          expected_gini_cnt_low) {
        for (int i = 0; i < n_classes; ++i) {
          printf("gini count class %u: %u/%u\n", i,
                 gini_cnt[res_indexes[l] * 2 * n_classes + i],
                 leaf_end_index[l] - leaf_start_index[l]);
#ifdef PRINT_ERROR_NO_ASSERT
          if (gini_cnt[res_indexes[l] * 2 * n_classes + i] !=
              expected_gini_cnt_low[l * n_classes + i])
            printf("ASSERT: gini cnt leaf %u class %u found %u expected %u\n",
                   l, i, gini_cnt[res_indexes[l] * 2 * n_classes + i],
                   expected_gini_cnt_low[l * n_classes + i]);
          if ((leaf_end_index[l] - leaf_start_index[l]) != expected_leaf_cnt[l])
            printf("ASSERT: leaf %u size found %u expected %u\n", l,
                   (leaf_end_index[l] - leaf_start_index[l]),
                   expected_leaf_cnt[l]);
#else
          assert(gini_cnt[res_indexes[l] * 2 * n_classes + i] ==
                 expected_gini_cnt_low[l * n_classes + i]);
          assert((leaf_end_index[l] - leaf_start_index[l]) ==
                 expected_leaf_cnt[l]);
#endif
        }
      }

      if (l >= start_n_leaves || cmds_array[l].type == SPLIT_COMMIT) {
        if (expected_feature_values) {
          uint32_t n_points_align = ((n_points + 1) >> 1) << 1;
          printf("split feature value:\n");
          for (int i = leaf_start_index[l]; i < leaf_end_index[l]; ++i) {
            printf("feature %u: %f\n", split_feature,
                   t_features[split_feature * n_points_align + i]);
#ifdef PRINT_ERROR_NO_ASSERT
            if (t_features[split_feature * n_points_align + i] !=
                expected_feature_values[split_feature * n_points + i])
              printf("ASSERT: feature value at index %u found %f expected %f\n",
                     i, t_features[split_feature * n_points_align + i],
                     expected_feature_values[split_feature * n_points + i]);
#else
            assert(t_features[split_feature * n_points_align + i] ==
                   expected_feature_values[split_feature * n_points + i]);
#endif
          }
          printf("other features value:\n");
          for (uint32_t f = 0; f < n_features; ++f) {
            if (f == split_feature)
              continue;
            for (int i = leaf_start_index[l]; i < leaf_end_index[l]; ++i) {
              printf("feature %u: %f\n", f, t_features[f * n_points_align + i]);
#ifdef PRINT_ERROR_NO_ASSERT
              if (t_features[f * n_points_align + i] !=
                  expected_feature_values[f * n_points + i])
                printf(
                    "ASSERT: feature value at index %u found %f expected %f\n",
                    i, t_features[f * n_points_align + i],
                    expected_feature_values[f * n_points + i]);
#else
              assert(t_features[f * n_points_align + i] ==
                     expected_feature_values[f * n_points + i]);
#endif
            }
          }
        }
        if (expected_target_values) {
          printf("targets value:\n");
          for (int i = leaf_start_index[l]; i < leaf_end_index[l]; ++i) {
            printf("target: %f\n", t_targets[i]);
#ifdef PRINT_ERROR_NO_ASSERT
            if (t_targets[i] != expected_target_values[i])
              printf("ASSERT: class value at index %u found %f expected %f\n",
                     i, t_targets[i], expected_target_values[i]);
#else
            assert(t_targets[i] == expected_target_values[i]);
#endif
          }
        }
      }

      if (l < start_n_leaves && cmds_array[l].type == SPLIT_MINMAX &&
          expected_min_max) {

        printf("min/max: %f/%f\n", min_max_feature[res_indexes[l] * 2],
               min_max_feature[res_indexes[l] * 2 + 1]);
#ifdef PRINT_ERROR_NO_ASSERT
        if (min_max_feature[res_indexes[l] * 2] != expected_min_max[l * 2])
          printf("ASSERT: min value leaf %u found %f expected %f\n", l,
                 min_max_feature[res_indexes[l] * 2], expected_min_max[l * 2]);
#else
        assert(min_max_feature[res_indexes[l] * 2] == expected_min_max[l * 2]);
#endif
#ifdef PRINT_ERROR_NO_ASSERT
        if (min_max_feature[res_indexes[l] * 2 + 1] !=
            expected_min_max[l * 2 + 1])
          printf("ASSERT: max value leaf %u found %f expected %f\n", l,
                 min_max_feature[res_indexes[l] * 2 + 1],
                 expected_min_max[l * 2 + 1]);
#else
        assert(min_max_feature[res_indexes[l] * 2 + 1] ==
               expected_min_max[l * 2 + 1]);
#endif
      }
    }
  }
  printf("--------------- TEST END -------------------\n");
}

#endif
