#include <libavcodec/avcodec.h>
#include <stdio.h>
int main() { printf("AC3=%d, EAC3=%d\n", AV_CODEC_ID_AC3, AV_CODEC_ID_EAC3); return 0; }
