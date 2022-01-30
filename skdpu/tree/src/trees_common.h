/**
 * @file common.h
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Common data structures and info for host and DPU
 *
 */

#ifndef _H_TREES_COMMON
#define _H_TREES_COMMON /**< header guard */

#include <stdint.h>

/** @name Debug defines
 */
/**@{*/
//#define DEBUG /**< debug switch for the C code (host and DPU) */
/**@}*/

/** @name Constraints
 * @brief Data size constraints
 */
/**@{*/
#define MAX_FEATURE_DPU                                                        \
  5000000 /**< How many features we fit into one DPU's MRAM. Can be            \
             increased further. */
#define MAX_SAMPLES_DPU                                                        \
  5000000 /**< How many samples we fit into one DPU's MRAM. Can be increased   \
             further. */
#define MAX_STACK_DPU                                                          \
  500 /**< How many split records we fit into one DPU's MRAM. Can be           \
         increased further. */
#define MAX_CLASSES 32 /**< How many different target classes we allow */
#define MAX_NB_LEAF                                                            \
  1024 /**< How many tree leaves are supported by the DPU program */
/**@}*/

// Define the size of features (choose one):

typedef float feature_t;
////////// OR
// typedef int8_t int_feature;
////////// OR
// typedef int16_t int_feature;
////////// OR
// typedef int32_t int_feature;
// #define FEATURETYPE_32

/**
 * @struct Command
 * @brief structure to store a command from the host for the DPU
 **/
#define SPLIT_EVALUATE 0
#define SPLIT_COMMIT 1
#define SPLIT_MINMAX 2
struct Command {

  uint8_t type; /**< type of command (split_evaluate=0 or split_commit=1)*/
  uint8_t feature_index;       /**< feature index for the split */
  uint16_t leaf_index;         /**< leaf index for the split */
  feature_t feature_threshold; /**< threshold for the split */
};

#define ALIGN_8_LOW(x) (((x) >> 3) << 3)
#define ALIGN_8_HIGH(x) (((x + 7) >> 3) << 3)

#endif
