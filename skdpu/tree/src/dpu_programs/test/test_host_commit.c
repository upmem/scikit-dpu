#include "../../trees.h"
#include <assert.h>
#include <dpu.h>
#include <stdint.h>
#include <stdio.h>

#define DPU_BINARY "trees_dpu_kernel_v2"

feature_t **read_features(FILE *features_file, uint32_t n_points) {

  feature_t **features = calloc(n_points, sizeof(feature_t *));
  for (uint32_t i = 0; i < n_points; ++i) {
    features[i] = calloc(4, sizeof(feature_t));
  }
  uint32_t p = 0;
  while (fscanf(features_file, "%f %f %f %f\n", &features[p][0],
                &features[p][1], &features[p][2], &features[p][3]) != EOF) {
    p++;
  }
  assert(p == n_points);

  return features;
}

feature_t *read_targets(FILE *targets_file, uint32_t n_points) {

  feature_t *targets = calloc(n_points, sizeof(feature_t));

  uint32_t p = 0;
  while (fscanf(targets_file, "%f\n", &targets[p]) != EOF) {
    p++;
  }
  assert(p == n_points);

  return targets;
}

/*#define PRINT_ONLY_NO_ASSERT*/

void check_gini_count(struct CommandResults *res,
                      uint32_t *expected_gini_count) {

  assert(res->nb_gini == 1);
  printf("gini count class 0 low: %u expected: %u\n", res->gini_cnt[0],
         expected_gini_count[0]);
  printf("gini count class 0 high: %u expected: %u\n", res->gini_cnt[3],
         expected_gini_count[1]);
  printf("gini count class 1 low: %u expected: %u\n", res->gini_cnt[1],
         expected_gini_count[2]);
  printf("gini count class 1 high: %u expected: %u\n", res->gini_cnt[4],
         expected_gini_count[3]);
  printf("gini count class 2 low: %u expected: %u\n", res->gini_cnt[2],
         expected_gini_count[4]);
  printf("gini count class 2 high: %u expected: %u\n", res->gini_cnt[5],
         expected_gini_count[5]);

#ifndef PRINT_ONLY_NO_ASSERT
  assert(res->gini_cnt[0] == expected_gini_count[0]);
  assert(res->gini_cnt[3] == expected_gini_count[1]);
  assert(res->gini_cnt[1] == expected_gini_count[2]);
  assert(res->gini_cnt[4] == expected_gini_count[3]);
  assert(res->gini_cnt[2] == expected_gini_count[4]);
  assert(res->gini_cnt[5] == expected_gini_count[5]);
#endif
}

/**
 * Test of several SPLIT_COMMIT commands
 * a SPLIT_EVALUATE is done after the commit, and
 * we check the result of the SPLIT_EVALUATE to assert
 * the functional correctness of the commit
 **/
int main() {

  struct Params tree_params;
  tree_params.nfeatures = 4;
  tree_params.npoints = 150;
  tree_params.ntargets = 3;
  tree_params.ndpu = 1;

  allocate(&tree_params);
  load_kernel(&tree_params, DPU_BINARY);

  FILE *features_file = fopen("test/data/X.txt", "r");
  FILE *targets_file = fopen("test/data/y.txt", "r");
  assert(features_file);
  assert(targets_file);
  feature_t **features = read_features(features_file, tree_params.npoints);
  feature_t *targets = read_targets(targets_file, tree_params.npoints);
  populateDpu(&tree_params, features, targets);

  struct CommandArray cmd_arr;
  struct Command cmd1;
  struct CommandResults res;

  /***** first commit *****/
  cmd1.type = SPLIT_COMMIT;
  cmd1.feature_index = 2;
  cmd1.leaf_index = 0;
  cmd1.feature_threshold = 4.848565;

  addCommand(&cmd_arr, cmd1);
  pushCommandArray(&tree_params, &cmd_arr);
  syncCommandArrayResults(&tree_params, &cmd_arr, &res);

  /***** end first commit *****/

  /*struct dpu_set_t dpu;*/
  /*DPU_FOREACH(tree_params.allset, dpu) {*/
    /*dpu_log_read(dpu, stdout);*/
  /*}*/

  /***** second commit *****/
  
  cmd1.type = SPLIT_COMMIT;
  cmd1.feature_index = 0;
  cmd1.leaf_index = 0;
  cmd1.feature_threshold = 6.377749;

  cmd_arr.nb_cmds = 0;
  addCommand(&cmd_arr, cmd1);
  pushCommandArray(&tree_params, &cmd_arr);
  syncCommandArrayResults(&tree_params, &cmd_arr, &res);

  /***** end second commit *****/

  /*DPU_FOREACH(tree_params.allset, dpu) {*/
    /*dpu_log_read(dpu, stdout);*/
  /*}*/

  /***** third commit *****/
  
  cmd1.type = SPLIT_COMMIT;
  cmd1.feature_index = 1;
  cmd1.leaf_index = 0;
  cmd1.feature_threshold = 4.014110;

  cmd_arr.nb_cmds = 0;
  addCommand(&cmd_arr, cmd1);
  pushCommandArray(&tree_params, &cmd_arr);
  syncCommandArrayResults(&tree_params, &cmd_arr, &res);

  /***** end third commit *****/

  /*DPU_FOREACH(tree_params.allset, dpu) {*/
    /*dpu_log_read(dpu, stdout);*/
  /*}*/

  /***** first split *****/

  cmd1.type = SPLIT_EVALUATE;
  cmd1.feature_index = 0;
  cmd1.leaf_index = 1;
  cmd1.feature_threshold = 6.950428788593854;

  cmd_arr.nb_cmds = 0;
  addCommand(&cmd_arr, cmd1);
  pushCommandArray(&tree_params, &cmd_arr);
  syncCommandArrayResults(&tree_params, &cmd_arr, &res);

  uint32_t expected_gini1[6] = {0, 0, 4, 0, 35, 12};
  check_gini_count(&res, expected_gini1);

  /***** end first split *****/

  /*DPU_FOREACH(tree_params.allset, dpu) {*/
    /*dpu_log_read(dpu, stdout);*/
  /*}*/

  /***** second split *****/

  cmd1.type = SPLIT_EVALUATE;
  cmd1.feature_index = 1;
  cmd1.leaf_index = 3;
  cmd1.feature_threshold = 4.167411513272327;

  cmd_arr.nb_cmds = 0;
  addCommand(&cmd_arr, cmd1);
  pushCommandArray(&tree_params, &cmd_arr);
  syncCommandArrayResults(&tree_params, &cmd_arr, &res);

  uint32_t expected_gini2[6] = {1, 2, 0, 0, 0, 0};
  check_gini_count(&res, expected_gini2);

  /***** end second split *****/

  /*DPU_FOREACH(tree_params.allset, dpu) {*/
    /*dpu_log_read(dpu, stdout);*/
  /*}*/

  /***** third split *****/

  cmd1.type = SPLIT_EVALUATE;
  cmd1.feature_index = 2;
  cmd1.leaf_index = 3;
  cmd1.feature_threshold = 1.4751792133281754;

  cmd_arr.nb_cmds = 0;
  addCommand(&cmd_arr, cmd1);
  pushCommandArray(&tree_params, &cmd_arr);
  syncCommandArrayResults(&tree_params, &cmd_arr, &res);

  uint32_t expected_gini3[6] = {1, 2, 0, 0, 0, 0};
  check_gini_count(&res, expected_gini3);

  /***** end third split *****/

  /*DPU_FOREACH(tree_params.allset, dpu) {*/
    /*dpu_log_read(dpu, stdout);*/
  /*}*/

  /***** fourth split *****/

  cmd1.type = SPLIT_EVALUATE;
  cmd1.feature_index = 3;
  cmd1.leaf_index = 1;
  cmd1.feature_threshold = 1.6900613370821986;

  cmd_arr.nb_cmds = 0;
  addCommand(&cmd_arr, cmd1);
  pushCommandArray(&tree_params, &cmd_arr);
  syncCommandArrayResults(&tree_params, &cmd_arr, &res);

  uint32_t expected_gini4[6] = {0, 0, 3, 1, 4, 43};
  check_gini_count(&res, expected_gini4);

  /***** end fourth split *****/

  /*DPU_FOREACH(tree_params.allset, dpu) {*/
    /*dpu_log_read(dpu, stdout);*/
  /*}*/

  return 0;
}
