#pragma once

#include <array>
#include <cstdint>

namespace stableunclip_autogen {

inline constexpr std::uint64_t kSpecFingerprint = 536003522699097492ULL;
inline constexpr const char *kProfileName = "stableunclip_autogen";

inline constexpr std::uint64_t kImageH = 4ULL;
inline constexpr std::uint64_t kImageW = 4ULL;
inline constexpr std::uint64_t kImageC = 1ULL;
inline constexpr std::uint64_t kDecodedH = 4ULL;
inline constexpr std::uint64_t kDecodedW = 4ULL;
inline constexpr std::uint64_t kDecodedC = 1ULL;
inline constexpr std::uint64_t kLatentH = 4ULL;
inline constexpr std::uint64_t kLatentW = 4ULL;
inline constexpr std::uint64_t kLatentC = 1ULL;
inline constexpr std::uint64_t kVaeScaleFactor = 1ULL;

inline constexpr std::uint64_t kVisionImageSize = 4ULL;
inline constexpr std::uint64_t kVisionPatchSize = 2ULL;
inline constexpr std::uint64_t kVisionWidth = 1024ULL;
inline constexpr std::uint64_t kVisionLayers = 32ULL;
inline constexpr std::uint64_t kVisionHeads = 16ULL;
inline constexpr std::uint64_t kVisionEmbedDim = 1024ULL;
inline constexpr std::uint64_t kVisionFfInner = 4096ULL;

inline constexpr std::uint64_t kTextVocabSize = 49408ULL;
inline constexpr std::uint64_t kTextHiddenSize = 768ULL;
inline constexpr std::uint64_t kTextIntermediateSize = 3072ULL;
inline constexpr std::uint64_t kTextLayers = 12ULL;
inline constexpr std::uint64_t kTextHeads = 12ULL;
inline constexpr std::uint64_t kTextSeqLen = 1ULL;

inline constexpr std::uint64_t kFeatureInChannels = 1024ULL;
inline constexpr std::uint64_t kFeatureHiddenDim = 1536ULL;
inline constexpr std::uint64_t kFeatureOutChannels = 768ULL;
inline constexpr std::uint64_t kFeatureLayers = 2ULL;

inline constexpr std::uint64_t kUnetInChannels = 1ULL;
inline constexpr std::uint64_t kUnetOutChannels = 1ULL;
inline constexpr std::uint64_t kUnetLayersPerBlock = 2ULL;
inline constexpr std::uint64_t kUnetAttentionHeadDim = 8ULL;
inline constexpr std::uint64_t kUnetCrossAttentionDim = 768ULL;
inline constexpr std::uint64_t kUnetTimeProjDim = 320ULL;
inline constexpr std::uint64_t kUnetTimeEmbedDim = 320ULL;
inline constexpr std::array<std::uint64_t, 4> kUnetBlockOutChannels{{ 320, 640, 1280, 1280 }};
inline constexpr std::array<std::uint64_t, 4> kUnetDownResnetCounts{{ 2, 2, 2, 2 }};
inline constexpr std::array<std::uint64_t, 4> kUnetDownTransformerBlocks{{ 2, 2, 2, 2 }};
inline constexpr std::array<std::uint64_t, 4> kUnetUpResnetCounts{{ 3, 3, 3, 3 }};
inline constexpr std::array<std::uint64_t, 4> kUnetUpTransformerBlocks{{ 3, 3, 3, 3 }};
inline constexpr std::uint64_t kUnetMidResnetCount = 2ULL;
inline constexpr std::uint64_t kUnetMidTransformerBlocks = 1ULL;
inline constexpr std::uint64_t kDefaultTransformerBlocks = 1ULL;

inline constexpr std::array<std::uint64_t, 4> kVaeBlockOutChannels{{ 128, 256, 512, 512 }};
inline constexpr std::uint64_t kVaeLayersPerBlock = 2ULL;
inline constexpr std::uint64_t kVaeMidChannels = 512ULL;
inline constexpr std::uint64_t kVaeNormNumGroups = 32ULL;

inline constexpr std::uint64_t kSuperresInChannels = 1ULL;
inline constexpr std::uint64_t kSuperresOutChannels = 1ULL;
inline constexpr std::uint64_t kSuperresHiddenChannels = 1280ULL;
inline constexpr std::uint64_t kSuperresLayers = 4ULL;

inline constexpr std::uint64_t kNumTrainTimesteps = 1000ULL;
inline constexpr double kBetaStart = 0.00085;
inline constexpr double kBetaEnd = 0.012;
inline constexpr std::uint64_t kGenerateNumInferenceSteps = 50ULL;
inline constexpr double kGenerateGuidanceScale = 7.5;
inline constexpr std::uint64_t kBatch = 1ULL;
inline constexpr std::uint64_t kClassLabelsDim = 1536ULL;
inline constexpr std::uint64_t kNoiseLevel = 0ULL;
inline constexpr std::uint64_t kCfgCopies = 1ULL;
inline constexpr std::uint64_t kDefaultVaeTembDim = 16ULL;
inline constexpr bool kUseUnetClassProj = true;
inline constexpr bool kUseLinearSchedulerFallback = false;
inline constexpr bool kFastMode = false;
inline constexpr bool kUnetCrossAttentionEnabled = true;
inline constexpr bool kUnetMidAttentionEnabled = true;
inline constexpr bool kTextEncoderEnabled = true;
inline constexpr bool kUseTextConditioning = true;
inline constexpr bool kFullSecureImageEncoderEnabled = true;
inline constexpr bool kVisionEncoderEnabled = true;
inline constexpr bool kVaeMidAttentionEnabled = true;
inline constexpr bool kEnableSuperres = false;
inline constexpr bool kSecurePromptLookup = true;
inline constexpr bool kUseLightImageEncoderFallback = false;
inline constexpr bool kTextTransformerEnabled = true;
inline constexpr bool kVisionTransformerEnabled = true;
inline constexpr bool kUnetTransformerEnabled = true;
inline constexpr bool kUnetMidAttnRuntimeEnabled = true;
inline constexpr double kOutputTanhGain = 1.0;
inline constexpr bool kOutputAutoscale = true;
inline constexpr const char *kTraceTemplateJsonl =
    "{\"name\": \"vision.patch_embedding.conv\", \"op\": \"conv\", \"phase\": \"vision\", \"shape_expr\": \"[B, 224, 224, 3] -> [B, 256, 1024]\", \"count_expr\": \"1\", \"notes\": \"\"}\n"
    "{\"name\": \"vision.pre_layernorm\", \"op\": \"layernorm\", \"phase\": \"vision\", \"shape_expr\": \"[B, 257, 1024]\", \"count_expr\": \"1\", \"notes\": \"\"}\n"
    "{\"name\": \"vision.block.attn.softmax\", \"op\": \"softmax\", \"phase\": \"vision\", \"shape_expr\": \"[B, 16, 257, 257]\", \"count_expr\": \"32\", \"notes\": \"\"}\n"
    "{\"name\": \"vision.block.mlp.gelu\", \"op\": \"gelu\", \"phase\": \"vision\", \"shape_expr\": \"[B, 257, 4096]\", \"count_expr\": \"32\", \"notes\": \"\"}\n"
    "{\"name\": \"vision.post_layernorm\", \"op\": \"layernorm\", \"phase\": \"vision\", \"shape_expr\": \"[B, 257, 1024]\", \"count_expr\": \"1\", \"notes\": \"\"}\n"
    "{\"name\": \"text.block.attn.softmax\", \"op\": \"softmax\", \"phase\": \"text\", \"shape_expr\": \"[B, 12, 77, 77]\", \"count_expr\": \"12\", \"notes\": \"\"}\n"
    "{\"name\": \"text.block.mlp.gelu\", \"op\": \"gelu\", \"phase\": \"text\", \"shape_expr\": \"[B, 77, 3072]\", \"count_expr\": \"12\", \"notes\": \"\"}\n"
    "{\"name\": \"text.final_layernorm\", \"op\": \"layernorm\", \"phase\": \"text\", \"shape_expr\": \"[B, 77, 768]\", \"count_expr\": \"1\", \"notes\": \"\"}\n"
    "{\"name\": \"feature_upscaler.gelu\", \"op\": \"gelu\", \"phase\": \"feature_upscaler\", \"shape_expr\": \"[B, 3072]\", \"count_expr\": \"4\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.time_embedding.exp\", \"op\": \"time_embedding_exp\", \"phase\": \"unet\", \"shape_expr\": \"[160]\", \"count_expr\": \"1 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.time_embedding.silu\", \"op\": \"silu\", \"phase\": \"unet\", \"shape_expr\": \"[B, 320]\", \"count_expr\": \"1 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.conv_in\", \"op\": \"conv\", \"phase\": \"unet\", \"shape_expr\": \"[B, 64, 64, 4] -> [B, 64, 64, 320]\", \"count_expr\": \"1 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.down0.res.groupnorm\", \"op\": \"groupnorm\", \"phase\": \"unet\", \"shape_expr\": \"[B, H0, W0, 320]\", \"count_expr\": \"2 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.down0.res.silu\", \"op\": \"silu\", \"phase\": \"unet\", \"shape_expr\": \"[B, H0, W0, 320]\", \"count_expr\": \"2 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.down0.attn.softmax\", \"op\": \"softmax\", \"phase\": \"unet\", \"shape_expr\": \"[B, heads=40, tokens=H0*W0, ctx=tokens or cond_len]\", \"count_expr\": \"4 per denoise step\", \"notes\": \"self-attention and cross-attention\"}\n"
    "{\"name\": \"unet.down1.res.groupnorm\", \"op\": \"groupnorm\", \"phase\": \"unet\", \"shape_expr\": \"[B, H1, W1, 640]\", \"count_expr\": \"2 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.down1.res.silu\", \"op\": \"silu\", \"phase\": \"unet\", \"shape_expr\": \"[B, H1, W1, 640]\", \"count_expr\": \"2 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.down1.attn.softmax\", \"op\": \"softmax\", \"phase\": \"unet\", \"shape_expr\": \"[B, heads=80, tokens=H1*W1, ctx=tokens or cond_len]\", \"count_expr\": \"4 per denoise step\", \"notes\": \"self-attention and cross-attention\"}\n"
    "{\"name\": \"unet.down2.res.groupnorm\", \"op\": \"groupnorm\", \"phase\": \"unet\", \"shape_expr\": \"[B, H2, W2, 1280]\", \"count_expr\": \"2 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.down2.res.silu\", \"op\": \"silu\", \"phase\": \"unet\", \"shape_expr\": \"[B, H2, W2, 1280]\", \"count_expr\": \"2 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.down2.attn.softmax\", \"op\": \"softmax\", \"phase\": \"unet\", \"shape_expr\": \"[B, heads=160, tokens=H2*W2, ctx=tokens or cond_len]\", \"count_expr\": \"4 per denoise step\", \"notes\": \"self-attention and cross-attention\"}\n"
    "{\"name\": \"unet.down3.res.groupnorm\", \"op\": \"groupnorm\", \"phase\": \"unet\", \"shape_expr\": \"[B, H3, W3, 1280]\", \"count_expr\": \"2 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.down3.res.silu\", \"op\": \"silu\", \"phase\": \"unet\", \"shape_expr\": \"[B, H3, W3, 1280]\", \"count_expr\": \"2 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.down3.attn.softmax\", \"op\": \"softmax\", \"phase\": \"unet\", \"shape_expr\": \"[B, heads=160, tokens=H3*W3, ctx=tokens or cond_len]\", \"count_expr\": \"4 per denoise step\", \"notes\": \"self-attention and cross-attention\"}\n"
    "{\"name\": \"unet.mid.groupnorm\", \"op\": \"layernorm\", \"phase\": \"unet\", \"shape_expr\": \"[B, 4096, 1280]\", \"count_expr\": \"1 per transformer norm site per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.mid.attn.softmax\", \"op\": \"softmax\", \"phase\": \"unet\", \"shape_expr\": \"[B, heads=160, 4096, 4096]\", \"count_expr\": \"2 per denoise step\", \"notes\": \"self-attention and cross-attention inside the mid transformer block\"}\n"
    "{\"name\": \"unet.up0.res.groupnorm\", \"op\": \"groupnorm\", \"phase\": \"unet\", \"shape_expr\": \"[B, Hup0, Wup0, 1280]\", \"count_expr\": \"3 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.up0.res.silu\", \"op\": \"silu\", \"phase\": \"unet\", \"shape_expr\": \"[B, Hup0, Wup0, 1280]\", \"count_expr\": \"3 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.up0.attn.softmax\", \"op\": \"softmax\", \"phase\": \"unet\", \"shape_expr\": \"[B, heads=160, tokens=Hup0*Wup0, ctx=tokens or cond_len]\", \"count_expr\": \"6 per denoise step\", \"notes\": \"self-attention and cross-attention\"}\n"
    "{\"name\": \"unet.up1.res.groupnorm\", \"op\": \"groupnorm\", \"phase\": \"unet\", \"shape_expr\": \"[B, Hup1, Wup1, 1280]\", \"count_expr\": \"3 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.up1.res.silu\", \"op\": \"silu\", \"phase\": \"unet\", \"shape_expr\": \"[B, Hup1, Wup1, 1280]\", \"count_expr\": \"3 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.up1.attn.softmax\", \"op\": \"softmax\", \"phase\": \"unet\", \"shape_expr\": \"[B, heads=160, tokens=Hup1*Wup1, ctx=tokens or cond_len]\", \"count_expr\": \"6 per denoise step\", \"notes\": \"self-attention and cross-attention\"}\n"
    "{\"name\": \"unet.up2.res.groupnorm\", \"op\": \"groupnorm\", \"phase\": \"unet\", \"shape_expr\": \"[B, Hup2, Wup2, 640]\", \"count_expr\": \"3 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.up2.res.silu\", \"op\": \"silu\", \"phase\": \"unet\", \"shape_expr\": \"[B, Hup2, Wup2, 640]\", \"count_expr\": \"3 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.up2.attn.softmax\", \"op\": \"softmax\", \"phase\": \"unet\", \"shape_expr\": \"[B, heads=80, tokens=Hup2*Wup2, ctx=tokens or cond_len]\", \"count_expr\": \"6 per denoise step\", \"notes\": \"self-attention and cross-attention\"}\n"
    "{\"name\": \"unet.up3.res.groupnorm\", \"op\": \"groupnorm\", \"phase\": \"unet\", \"shape_expr\": \"[B, Hup3, Wup3, 320]\", \"count_expr\": \"3 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.up3.res.silu\", \"op\": \"silu\", \"phase\": \"unet\", \"shape_expr\": \"[B, Hup3, Wup3, 320]\", \"count_expr\": \"3 per denoise step\", \"notes\": \"\"}\n"
    "{\"name\": \"unet.up3.attn.softmax\", \"op\": \"softmax\", \"phase\": \"unet\", \"shape_expr\": \"[B, heads=40, tokens=Hup3*Wup3, ctx=tokens or cond_len]\", \"count_expr\": \"6 per denoise step\", \"notes\": \"self-attention and cross-attention\"}\n"
    "{\"name\": \"vae.decode\", \"op\": \"reconstruct\", \"phase\": \"vae\", \"shape_expr\": \"[B, 64, 64, 4] -> image\", \"count_expr\": \"1\", \"notes\": \"\"}\n"
    "{\"name\": \"superres.resblock.silu\", \"op\": \"silu\", \"phase\": \"superres\", \"shape_expr\": \"[B, H, W, 1280]\", \"count_expr\": \"4\", \"notes\": \"\"}\n"
    "";

}  // namespace stableunclip_autogen
