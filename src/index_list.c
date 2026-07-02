#include "index_list.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Add a new node to the linked list.
///
void index_list_add_entry(struct index_list* list, const glong index)
{
	struct index_list_node* new_node = aligned_malloc(sizeof(new_node[0]));
	new_node->data = index;
	new_node->next = list->head;

	list->head = new_node;
	list->size++;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy the entries of the list into a linear array.
///
void index_list_to_array(const struct index_list* list, glong* entries)
{
	const struct index_list_node* node = list->head;
	glong i = 0;
	while (node != NULL)
	{
		entries[i] = node->data;
		node = node->next;
		i++;
	}
	assert(i == list->size);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the linked list (free memory).
///
void delete_index_list(struct index_list* list)
{
	while (list->head != NULL)
	{
		struct index_list_node* next = list->head->next;
		aligned_free(list->head);
		list->size--;
		list->head = next;
	}
	assert(list->size == 0);
}
