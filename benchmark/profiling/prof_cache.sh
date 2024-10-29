#

perf stat -B -e cache-references,cache-misses,cycles,instructions,branches $1
