#ifndef SHIM_H
#define SHIM_H

#include <stddef.h>
#include <stdint.h>


/* Task type contains necessary parameters for executing the program. */
typedef struct Task {
  const char *id;
  const char *const *args;
  const char *const *files;
} Task;

/* new_task_result is a function to construct the task result type. It takes
 * result data as a parameter, in case there is some. */
void *new_task_result(uint8_t *data, size_t len);

/* add_file_to_result allows adding persisted files to task result for
 * returning back to Gevulot node. */
void add_file_to_result(void *result, const char *file_name);

/* run function is the entry point for Gevulot. It takes callback function
 * pointer as a parameter and invokes it when there is task for execution.
 * The callback function is then expected to return task result once it's
 * finished. */
void run(void *(*callback)(const struct Task*));

#endif
