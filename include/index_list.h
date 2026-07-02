#pragma once

#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Linked list node for storing indices.
///
struct index_list_node
{
	glong data;                    //!< index data entry
	struct index_list_node* next;  //!< pointer to next node
};


//________________________________________________________________________________________________________________________
///
/// \brief Linked list for storing indices.
///
struct index_list
{
	struct index_list_node* head;  //!< pointer to head node, NULL for an empty list
	glong size;                    //!< number of entries in the list
};


void index_list_add_entry(struct index_list* list, const glong index);


void index_list_to_array(const struct index_list* list, glong* entries);


void delete_index_list(struct index_list* list);
