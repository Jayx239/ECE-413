#ifndef MT_H_
#define MT_H_

#define NULL 0
typedef struct calc_t {
    int i;
    int j;
    int k;
    int type;   // 0: set to zero, 1: set to one, 2 division, 3 elimination,
    struct calc_t* next_node;
} calc_t;

calc_t* push(calc_t * head, int i, int j, int k, int type) {
    calc_t* next_node;
    if(head->i == -1) {
    head->i = i;
    head->j = j;
    head->k = k;
    head->type = type;
    head->next_node = NULL;
    return head;
    }
    next_node = (calc_t*) malloc(sizeof(calc_t));
    next_node->i = i;
    next_node->j = j;
    next_node->k = k;
    next_node->type = type;
    next_node->next_node = NULL; 
    
    calc_t* iter;
    iter = head;
    if(head != NULL ) {
        while(iter->next_node != NULL && iter->next_node->i != -1)
            iter = iter->next_node;
        iter->next_node = next_node;
    }
    else {
        head =  next_node;
    }
    return next_node;
}

calc_t * pop(calc_t* head) {
    
    calc_t* last_node = head;
    if(head!= NULL)
        head = head->next_node;
    return last_node;
}
#endif
