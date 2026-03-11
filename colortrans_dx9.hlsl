// colortrans3_dx9.hlsl
// MPC-HC / DX9 Pixel Shader port (ps_3_0)
//
// $MinimumShaderProfile: ps_3_0
//
// Notes:
// - This is a DX9-style pixel shader (sampler2D + tex2D + COLOR output).
// - MPC-HC DX9 pixel shaders typically don't provide custom constant buffers,
//   so this uses your baked defaults (edit below if desired).

sampler2D s0 : register(s0);

// -----------------------------------------------------------------------------
// Defaults (edit these if you want different “baked” tuning)
// -----------------------------------------------------------------------------
static const float  gamma_pow = 1.0;
static const float  lift      = -0.15;
static const float  gain      = 1.75;
static const float3 rgb_mult  = float3(1.0, 1.0, 1.0);

// mpv-style saturation default in range [-100 .. +100]
static const float  mpv_saturation_default = -22.5;

// -----------------------------------------------------------------------------
// Reverse ColorTrans constants (as per your DX11 shader)
// -----------------------------------------------------------------------------
#define REVERSE_COLORTRANS_ENABLE 1

static const float  colortrans_y_offset_10b  = 200.0;
static const float  colortrans_yoff_strength = 0.25; // try 0.10–0.30

static const float  colortrans_yoff =
    (colortrans_y_offset_10b / 1023.0) * colortrans_yoff_strength;

static const float  reverse_black_lift = 0.020;

// Precomputed inverse (RGB-domain), BT.709-style
static const float3 colortrans_inv_row0_709 = float3( 1.17866031,  0.17460893,  0.01571472);
static const float3 colortrans_inv_row1_709 = float3(-0.11147506,  1.55408099, -0.07362197);
static const float3 colortrans_inv_row2_709 = float3( 0.03243649,  0.11275820,  1.22378927);

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
float3 apply_saturation_mpv(float3 c, float mpv_sat)
{
    float sat = 1.0 + (mpv_sat / 100.0);
    sat = clamp(sat, 0.0, 3.0);

    // Rec.709 luma
    float y = dot(c, float3(0.2126, 0.7152, 0.0722));
    return lerp(y.xxx, c, sat);
}

float3 reverse_colortrans(float3 rgb_in)
{
    // Remove (part of) camera Y offset as neutral RGB bias
    float3 x = rgb_in - colortrans_yoff.xxx;

    float3 y;
    y.r = dot(colortrans_inv_row0_709, x);
    y.g = dot(colortrans_inv_row1_709, x);
    y.b = dot(colortrans_inv_row2_709, x);

    // Prevent black crush
    y += reverse_black_lift.xxx;
    return y;
}

float3 apply_grade(float3 c)
{
    // Match your DX11 behavior: clamp input before gamma
    c = saturate(c);

    float g = clamp(gamma_pow, 0.10, 5.0);
    float l = clamp(lift,     -1.0,  1.0);
    float k = clamp(gain,      0.0, 10.0);

    float3 m = clamp(rgb_mult, 0.0,  4.0);

    float3 y = pow(c, g);
    y += l;
    y *= k;
    y *= m;

    y = apply_saturation_mpv(y, mpv_saturation_default);
    return saturate(y);
}

// -----------------------------------------------------------------------------
// Pixel shader entry point (MPC-HC expects main())
// -----------------------------------------------------------------------------
float4 main(float2 tex : TEXCOORD0) : COLOR
{
    float4 src = tex2D(s0, tex);

    float3 c = src.rgb;

#if REVERSE_COLORTRANS_ENABLE
    c = reverse_colortrans(c);
#endif

    float3 out_rgb = apply_grade(c);
    return float4(out_rgb, src.a);
}
