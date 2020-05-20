#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "../headers/helper.h"

void printTime(struct timeval start, struct timeval end, const char* str){
    unsigned long ss,es,su,eu,s,u;
    ss  =start.tv_sec;
    su = start.tv_usec;
    es = end.tv_sec;
    eu = end.tv_usec;
    s = es - ss;
    if(eu > su){
        u = eu - su;
    }else{
        s--;
        u = 1000000 + eu - su;
    }
   
    printf("%s,%lu,%lu\n",str,s,u);
}
