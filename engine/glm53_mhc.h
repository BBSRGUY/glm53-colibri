#ifndef GLM53_MHC_H
#define GLM53_MHC_H
/* GLM-5.3 mHC (manifold-constrained hyper-connections) glue.
 *
 * Split out of glm53.c so tests/test_glm53_mhc.c exercises THE SAME code the engine
 * runs, rather than a transcription of it. The heavy lifting is dsv4_mhc.h (DeepSeek-V4,
 * unchanged); what lives here is the part GLM-5.3 pins down:
 *
 *   - the three constants dsv4_mhc_pre takes as parameters. From
 *     coli_v4_hc_split_sinkhorn (deepseek_v4.c): post is scaled by 2.0, and the pre and
 *     Sinkhorn epsilons are both hc_eps.
 *   - the stream SEED: broadcast, matching
 *     inputs_embeds.unsqueeze(2).expand(-1,-1,hc_mult,-1).
 *   - the stream COLLAPSE: arithmetic MEAN. This is GLM-5.3's one divergence from
 *     DeepSeek-V4, which collapses with a LEARNED head (DeepseekV4HyperHead). GLM-5.3
 *     ships no model-level hyper-connection tensors, and llama.cpp's glm5next graph
 *     collapses with glm5next_hc_mean.
 */
#include <string.h>
#include "dsv4_mhc.h"

/* post[M], comb[M*M], collapsed[D] <- residual[M*D] */
static inline void glm53_mhc_pre(const float *residual, const float *fn,
                                 const float *scale, const float *base,
                                 int M, int D, float rms_eps, float hc_eps, int iters,
                                 float *post, float *comb, float *collapsed){
    memset(collapsed, 0, (size_t)D*sizeof(float));   /* dsv4_mhc_pre accumulates into it */
    dsv4_mhc_pre(residual, fn, scale, base, M, D,
                 rms_eps, hc_eps, hc_eps, 2.0f, iters,
                 post, comb, collapsed);
}

/* out[M*D] <- post (x) blkout + comb^T @ residual.  Must NOT alias residual: every
 * output stream reads every input stream. */
static inline void glm53_mhc_post(const float *blkout, const float *residual,
                                  const float *post, const float *comb,
                                  int M, int D, float *out){
    dsv4_mhc_post(blkout, residual, post, comb, M, D, out);
}

/* [S,D] -> [S,M,D] by broadcast */
static inline void glm53_hc_seed(const float *embed, int S, int M, int D, float *out){
    for(int s=0;s<S;s++)
        for(int j=0;j<M;j++)
            memcpy(out+((size_t)s*M+j)*D, embed+(size_t)s*D, (size_t)D*sizeof(float));
}

/* [S,M,D] -> [S,D] by arithmetic mean over the stream axis */
static inline void glm53_hc_mean(const float *streams, int S, int M, int D, float *out){
    for(int s=0;s<S;s++){
        const float *src=streams+(size_t)s*M*D; float *dst=out+(size_t)s*D;
        for(int d=0;d<D;d++){
            float acc=0.f;
            for(int j=0;j<M;j++) acc+=src[(size_t)j*D+d];
            dst[d]=acc/(float)M;
        }
    }
}
#endif
