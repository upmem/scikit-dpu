/**
 * @file trees.h
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @author Julien Legriel (jlegriel@upmem.com)
 * @brief Implementation file for the trees project
 *
 */

#include "dpu.h"
#include "trees.h"
#include "trees_common.h"
#include <assert.h>

/**
 * @brief Allocates all DPUs
 *
 * @param p Algorithm parameters.
 */
void allocate(Params *p) {

  assert(p);
  DPU_ASSERT(dpu_alloc(p->ndpu, NULL, &(p->allset)));
}

/**
 * @brief Frees all DPUs
 *
 * @param p Algorithm parameters.
 */
void free_dpus(Params *p) {

  assert(p);
  DPU_ASSERT(dpu_free(p->allset));
}

/**
 * @brief Loads a binary in the DPUs.
 *
 * @param p Algorithm parameters.
 * @param DPU_BINARY path to the binary
 */
void load_kernel(Params *p, const char *DPU_BINARY) {

  assert(p);
  DPU_ASSERT(dpu_load(p->allset, DPU_BINARY, NULL));
}

struct callback_args {

  feature_t **points;
  feature_t *targets;
  uint32_t *start_index_dpu;
  uint32_t *nr_points_per_dpu;
  feature_t **dpu_features_buffer;
  uint32_t n_features;
  uint32_t *feature_index;
  uint32_t *point_index;
  uint32_t size_batch;
};

dpu_error_t dpu_rank_points_vector_callback(struct dpu_set_t rank,
                                            uint32_t rank_id, void *args) {

  struct callback_args *cb_args = ((struct callback_args **)args)[rank_id];
  uint32_t nr_dpus_rank;
  DPU_ASSERT(dpu_get_nr_dpus(rank, &nr_dpus_rank));

  bool is_target = true;
  for (uint32_t each_dpu = 0; each_dpu < nr_dpus_rank; ++each_dpu)
    is_target = cb_args->feature_index[each_dpu] == cb_args->n_features;

  for (uint32_t each_dpu = 0; each_dpu < nr_dpus_rank; ++each_dpu) {
    uint32_t batch_index = 0;
    while (batch_index < cb_args->size_batch) {
      if ((cb_args->feature_index[each_dpu] > cb_args->n_features) ||
          (!is_target &&
           (cb_args->feature_index[each_dpu] == cb_args->n_features))) {
        cb_args->dpu_features_buffer[each_dpu][batch_index++] = 0;
      } else if (cb_args->point_index[each_dpu] <
                 cb_args->nr_points_per_dpu[each_dpu]) {
        uint32_t next_point =
            cb_args->start_index_dpu[each_dpu] + cb_args->point_index[each_dpu];
        if (cb_args->feature_index[each_dpu] == cb_args->n_features) {
          assert(is_target);
          cb_args->dpu_features_buffer[each_dpu][batch_index++] =
              cb_args->targets[next_point];
        } else {
          cb_args->dpu_features_buffer[each_dpu][batch_index++] =
              cb_args->points[next_point][cb_args->feature_index[each_dpu]];
        }
        cb_args->point_index[each_dpu]++;
      } else {
        cb_args->point_index[each_dpu] = 0;
        cb_args->feature_index[each_dpu]++;
      }
    }
  }
  return DPU_OK;
}

#define SIZE_BATCH_POINT_TRANSFER (1024 * 256)

/**
 * @brief Fills the DPUs with their assigned points.
 */
void populateDpu(Params *p, feature_t **features, feature_t *targets) {

  uint32_t nr_ranks;
  DPU_ASSERT(dpu_get_nr_ranks(p->allset, &nr_ranks));
  struct callback_args *cb_args =
      calloc(nr_ranks, sizeof(struct callback_args));
  struct dpu_set_t rank, dpu;
  uint32_t each_rank, each_dpu;
  uint32_t nr_points_dpu = p->npoints / p->ndpu;
  uint32_t points_rest = p->npoints - (nr_points_dpu * p->ndpu);
  uint32_t start_index_dpu = 0;
  uint32_t dpu_index = 0;

  // prepare the data necessary for the callback
  DPU_RANK_FOREACH(p->allset, rank, each_rank) {
    cb_args[each_rank].points = features;
    cb_args[each_rank].targets = targets;
    cb_args[each_rank].n_features = p->nfeatures;
    uint32_t nr_dpus_rank;
    DPU_ASSERT(dpu_get_nr_dpus(rank, &nr_dpus_rank));
    cb_args[each_rank].start_index_dpu = calloc(nr_dpus_rank, sizeof(uint32_t));
    cb_args[each_rank].nr_points_per_dpu =
        calloc(nr_dpus_rank, sizeof(uint32_t));
    cb_args[each_rank].dpu_features_buffer =
        calloc(nr_dpus_rank, sizeof(feature_t *));
    cb_args[each_rank].feature_index = calloc(nr_dpus_rank, sizeof(uint32_t));
    cb_args[each_rank].point_index = calloc(nr_dpus_rank, sizeof(uint32_t));
    cb_args[each_rank].size_batch = SIZE_BATCH_POINT_TRANSFER;
    for (uint32_t i = 0; i < nr_dpus_rank; ++i) {
      cb_args[each_rank].start_index_dpu[i] = start_index_dpu;
      cb_args[each_rank].nr_points_per_dpu[i] =
          (dpu_index < points_rest) ? nr_points_dpu + 1 : nr_points_dpu;
      cb_args[each_rank].dpu_features_buffer[i] =
          calloc(SIZE_BATCH_POINT_TRANSFER, sizeof(feature_t));
      dpu_index++;
      start_index_dpu += cb_args[each_rank].nr_points_per_dpu[i];
    }
    DPU_FOREACH(rank, dpu, each_dpu) {
      DPU_ASSERT(dpu_prepare_xfer(
          dpu, cb_args[each_rank].dpu_features_buffer[each_dpu]));
    }
  }
  assert(start_index_dpu == p->npoints);

  // loop over all batches, first build the buffer of features for the DPU then
  // send it
  uint32_t nb_batches =
      ((p->npoints * p->nfeatures) + SIZE_BATCH_POINT_TRANSFER - 1) /
      SIZE_BATCH_POINT_TRANSFER;
  for (uint32_t i = 0; i < nb_batches; ++i) {

    DPU_ASSERT(dpu_callback(p->allset, dpu_rank_points_vector_callback,
                            &cb_args, DPU_CALLBACK_ASYNC));

    DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_TO_DPU, "t_features",
                             i * SIZE_BATCH_POINT_TRANSFER * sizeof(feature_t),
                             SIZE_BATCH_POINT_TRANSFER * sizeof(feature_t),
                             DPU_XFER_ASYNC));
  }

  // Now handle the targets
  nb_batches =
      (p->npoints + SIZE_BATCH_POINT_TRANSFER - 1) / SIZE_BATCH_POINT_TRANSFER;

  for (uint32_t i = 0; i < nb_batches; ++i) {

    DPU_ASSERT(dpu_callback(p->allset, dpu_rank_points_vector_callback,
                            &cb_args, DPU_CALLBACK_ASYNC));

    DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_TO_DPU, "t_targets",
                             i * SIZE_BATCH_POINT_TRANSFER * sizeof(feature_t),
                             SIZE_BATCH_POINT_TRANSFER * sizeof(feature_t),
                             DPU_XFER_ASYNC));
  }

  // transfer meta-data to DPU
  DPU_ASSERT(dpu_broadcast_to(p->allset, "n_points", 0, &(p->npoints),
                              sizeof(uint64_t), DPU_XFER_ASYNC));
  DPU_ASSERT(dpu_broadcast_to(p->allset, "n_features", 0, &(p->nfeatures),
                              sizeof(uint32_t), DPU_XFER_ASYNC));
  DPU_ASSERT(dpu_broadcast_to(p->allset, "n_classes", 0, &(p->ntargets),
                              sizeof(uint32_t), DPU_XFER_ASYNC));
  uint32_t nleaves = 1;
  uint32_t leaf_start_index = 0;
  uint32_t leaf_end_index = p->npoints;
  DPU_ASSERT(dpu_broadcast_to(p->allset, "n_leaves", 0, &nleaves,
                              sizeof(uint32_t), DPU_XFER_ASYNC));
  DPU_ASSERT(dpu_broadcast_to(p->allset, "leaf_start_index", 0,
                              &leaf_start_index, sizeof(uint32_t),
                              DPU_XFER_ASYNC));
  DPU_ASSERT(dpu_broadcast_to(p->allset, "leaf_end_index", 0, &leaf_end_index,
                              sizeof(uint32_t), DPU_XFER_ASYNC));

  // synchronize all ranks
  DPU_ASSERT(dpu_sync(p->allset));

  // free memory
  DPU_RANK_FOREACH(p->allset, rank, each_rank) {
    free(cb_args[each_rank].start_index_dpu);
    free(cb_args[each_rank].nr_points_per_dpu);
    free(cb_args[each_rank].feature_index);
    free(cb_args[each_rank].point_index);
    uint32_t nr_dpus_rank;
    DPU_ASSERT(dpu_get_nr_dpus(rank, &nr_dpus_rank));
    for (uint32_t i = 0; i < nr_dpus_rank; ++i) {
      free(cb_args[each_rank].dpu_features_buffer[i]);
    }
    free(cb_args[each_rank].dpu_features_buffer);
  }
  free(cb_args);
}

void addCommand(struct CommandArray *arr, struct Command cmd) {

  assert(arr->nb_cmds < MAX_NB_LEAF);
  arr->cmds[arr->nb_cmds++] = cmd;
}

void pushCommandArray(Params *p, struct CommandArray *arr) {

  // transfer the command array
  DPU_ASSERT(dpu_broadcast_to(p->allset, "cmds_array", 0, arr->cmds,
                              arr->nb_cmds * sizeof(struct Command),
                              DPU_XFER_ASYNC));
  DPU_ASSERT(dpu_broadcast_to(p->allset, "nb_cmds", 0, &(arr->nb_cmds),
                              sizeof(uint32_t), DPU_XFER_ASYNC));

  // launch the DPUs
  DPU_ASSERT(dpu_launch(p->allset, DPU_ASYNCHRONOUS));
}

void syncCommandArray(Params *p, struct CommandArray *arr,
                      struct CommandResults **res) {

  // first compute how many split evaluate and split min max commands were sent
  uint32_t nb_gini_cmds = 0, nb_min_max_cmds = 0;
  for (uint32_t i = 0; i < arr->nb_cmds; ++i) {
    if (arr->cmds[i].type == SPLIT_EVALUATE)
      nb_gini_cmds++;
    else if (arr->cmds[i].type == SPLIT_MINMAX)
      nb_min_max_cmds++;
  }

  // transfer results
  struct dpu_set_t dpu;
  uint32_t each_dpu;
  DPU_FOREACH(p->allset, dpu, each_dpu) {
    res[each_dpu]->nb_gini = nb_gini_cmds;
    DPU_ASSERT(dpu_prepare_xfer(dpu, res[each_dpu]->gini_cnt));
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_FROM_DPU, "gini_cnt", 0,
                           nb_gini_cmds * p->ntargets * sizeof(uint32_t),
                           DPU_XFER_ASYNC));
  DPU_FOREACH(p->allset, dpu, each_dpu) {
    res[each_dpu]->nb_minmax = nb_min_max_cmds;
    DPU_ASSERT(dpu_prepare_xfer(dpu, res[each_dpu]->min_max));
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_FROM_DPU, "min_max_feature", 0,
                           nb_min_max_cmds * 2 * sizeof(feature_t),
                           DPU_XFER_ASYNC));

  // sync all ranks
  DPU_ASSERT(dpu_sync(p->allset));
}
