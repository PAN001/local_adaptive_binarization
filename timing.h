#ifndef OPENALPR_TIMING_H
#define OPENALPR_TIMING_H

#include <iostream>
#include <ctime>
#include <stdint.h>
#include <sys/time.h>

// Support for OS X
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/clock.h>
#include <mach/mach.h>
#endif

void getTimeMonotonic(timespec* time);
int64_t getTimeMonotonicMs();

double diffclock(timespec time1,timespec time2);

int64_t getEpochTimeMs();