// VortexRT.fx
// Advanced rendering with SSS, reflections, AO, and color grading

//================== Includes ==================
#include "ReShade.fxh"
#include "ReShadeUI.fxh"

//================== Constants ==================
static const float PI = 3.14159265359;
static const float EPSILON = 0.0001;

// Quality-dependent sample counts
static const int SSS_SAMPLES[4] = { 4, 6, 8, 12 };
static const int AO_SAMPLES[4] = { 4, 8, 12, 16 };
static const int SSR_STEPS[4] = { 8, 16, 24, 32 };  // Reduced maximum steps

//================== Matrix Definitions ==================
uniform float4x4 ProjectionMatrix : PROJECTION;
uniform float4x4 ViewMatrix : VIEW;

//================== Textures ==================
texture2D texColor : COLOR;
texture2D texDepth : DEPTH;

// Intermediate textures
texture2D texNormal { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
texture2D texSpecular { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
texture2D texSSS { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
texture2D texMaterial { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
texture2D texSSR { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };

//================== Samplers ==================
sampler2D samplerColor { Texture = texColor; };
sampler2D samplerDepth { Texture = texDepth; };
sampler2D samplerNormal { Texture = texNormal; };
sampler2D samplerSpecular { Texture = texSpecular; };
sampler2D samplerSSS { Texture = texSSS; };
sampler2D samplerMaterial { Texture = texMaterial; };
sampler2D samplerSSR { Texture = texSSR; };

//================== UI Settings ==================
uniform int u_Quality_Mode <
    ui_type = "combo";
    ui_label = "Quality Mode";
    ui_items = "Low\0Medium\0High\0Ultra\0";
    ui_tooltip = "Adjusts overall effect quality and performance";
> = 1;

uniform bool u_Enable_SSS <
    ui_label = "Enable Subsurface Scattering";
    ui_tooltip = "Toggle SSS calculations";
> = true;

uniform bool u_Enable_SSR <
    ui_label = "Enable Screen Space Reflections";
    ui_tooltip = "Toggle SSR calculations";
> = true;

uniform bool u_Enable_AO <
    ui_label = "Enable Ambient Occlusion";
    ui_tooltip = "Toggle AO calculations";
> = true;

uniform bool u_Enable_Denoise <
    ui_label = "Enable Denoising";
    ui_tooltip = "Toggle reflection denoising";
> = true;

uniform float u_Exposure <
    ui_type = "slider";
    ui_label = "Exposure";
    ui_min = 0.0; ui_max = 5.0;
> = 1.0;

uniform float u_SSS_Strength <
    ui_type = "slider";
    ui_label = "Subsurface Scattering Strength";
    ui_min = 0.0; ui_max = 1.0;
> = 0.5;

uniform float u_Roughness <
    ui_type = "slider";
    ui_label = "Surface Roughness";
    ui_min = 0.0; ui_max = 1.0;
> = 0.5;

uniform float u_AO_Strength <
    ui_type = "slider";
    ui_label = "Ambient Occlusion Strength";
    ui_min = 0.0; ui_max = 4.0;
> = 1.0;

uniform float u_AO_Radius <
    ui_type = "slider";
    ui_label = "AO Radius";
    ui_min = 0.0; ui_max = 1.0;
> = 0.5;

uniform float u_SSR_Intensity <
    ui_type = "slider";
    ui_label = "SSR Intensity";
    ui_min = 0.0; ui_max = 1.0;
> = 0.5;

uniform float u_FresnelPower <
    ui_type = "slider";
    ui_label = "Fresnel Power";
    ui_min = 1.0; ui_max = 5.0;
> = 3.0;

uniform float u_Denoise_Strength <
    ui_type = "slider";
    ui_label = "Denoise Strength";
    ui_min = 0.0; ui_max = 1.0;
> = 0.5;

uniform float3 u_ColorTemp <
    ui_type = "color";
    ui_label = "Color Temperature";
> = float3(1.0, 1.0, 1.0);

uniform float u_Saturation <
    ui_type = "slider";
    ui_label = "Saturation";
    ui_min = 0.0; ui_max = 2.0;
> = 1.0;

uniform float u_Contrast <
    ui_type = "slider";
    ui_label = "Contrast";
    ui_min = 0.0; ui_max = 2.0;
> = 1.0;

//================== Helper Functions ==================
float3 GetScreenPosition(float2 texcoord, float depth)
{
    float3 screenPos;
    screenPos.xy = texcoord * 2.0f - 1.0f;
    screenPos.y = -screenPos.y;
    screenPos.z = depth;
    return screenPos;
}

float3 GetViewPosition(float2 texcoord, float depth)
{
    float3 screenPos = GetScreenPosition(texcoord, depth);
    float4 viewPos = mul(float4(screenPos, 1.0f), ViewMatrix);
    return viewPos.xyz / viewPos.w;
}

float3 GetNormals(float2 texcoord)
{
    float2 pixelSize = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
    float3 normal;
    
    float depth = tex2D(samplerDepth, texcoord).r;
    float depthLeft = tex2D(samplerDepth, texcoord - float2(pixelSize.x, 0)).r;
    float depthRight = tex2D(samplerDepth, texcoord + float2(pixelSize.x, 0)).r;
    float depthTop = tex2D(samplerDepth, texcoord - float2(0, pixelSize.y)).r;
    float depthBottom = tex2D(samplerDepth, texcoord + float2(0, pixelSize.y)).r;
    
    normal.x = depthLeft - depthRight;
    normal.y = depthBottom - depthTop;
    normal.z = 2.0 * pixelSize.x;
    
    return normalize(normal);
}

float GetFresnel(float3 normal, float3 viewDir)
{
    float NdotV = saturate(dot(normal, viewDir));
    return pow(1.0 - NdotV, u_FresnelPower);
}

float GetEdgeFactor(float2 texcoord, float depth, float3 normal)
{
    float2 pixelSize = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
    float depthDiff = 0.0;
    float normalDiff = 0.0;
    
    [unroll]
    for(int i = 0; i < 4; i++)
    {
        float angle = i * PI / 2.0;
        float2 offset = float2(cos(angle), sin(angle)) * pixelSize;
        
        float sampleDepth = tex2D(samplerDepth, texcoord + offset).r;
        float3 sampleNormal = tex2D(samplerNormal, texcoord + offset).rgb * 2.0 - 1.0;
        
        depthDiff += abs(depth - sampleDepth);
        normalDiff += 1.0 - saturate(dot(normal, sampleNormal));
    }
    
    return saturate(1.0 - (depthDiff + normalDiff) * 10.0);
}

float4 AnalyzeMaterial(float3 color, float3 normal, float depth)
{
    float4 material;
    
    // Estimate specularity from color variance
    float colorVariance = length(color - dot(color, float3(0.333, 0.333, 0.333)));
    
    // Estimate roughness from normal and depth variation
    float roughness = saturate(1.0 - abs(normal.z));
    
    // Detect skin tones (improved detection)
    float3 skinTone = normalize(float3(1.0, 0.8, 0.6));
    float skinLikelihood = saturate(dot(normalize(color), skinTone));
    skinLikelihood = smoothstep(0.5, 0.8, skinLikelihood);
    
    // Enhanced material properties
    material.r = saturate(1.0 - skinLikelihood + colorVariance * 0.5); // Reflectivity
    material.g = roughness; // Surface roughness
    material.b = colorVariance; // Color variation
    material.a = skinLikelihood; // Skin detection
    
    return material;
}

float3 CalculateSSS(float2 texcoord, float3 normal, float depth, float skinFactor)
{
    if(!u_Enable_SSS) return 0.0;
    
    float3 sssColor = 0;
    float sssRadius = 0.01 * u_SSS_Strength * (skinFactor + 0.2);
    
    int sampleCount = SSS_SAMPLES[u_Quality_Mode];
    float weight_sum = 0.0;
    
    [unroll]
    for(int i = 0; i < sampleCount; i++)
    {
        float angle = i * (2.0 * PI / sampleCount);
        float2 offset = float2(cos(angle), sin(angle)) * sssRadius;
        float2 sampleCoord = texcoord + offset;
        
        float3 sampleColor = tex2D(samplerColor, sampleCoord).rgb;
        float sampleDepth = tex2D(samplerDepth, sampleCoord).r;
        
        float weight = saturate(1.0 - abs(depth - sampleDepth) * 100.0);
        weight *= saturate(1.0 - length(offset) * 2.0);
        
        sssColor += sampleColor * weight;
        weight_sum += weight;
    }
    
    return sssColor / max(weight_sum, 0.001);
}

float CalculateAO(float2 texcoord, float3 normal, float depth)
{
    if(!u_Enable_AO) return 1.0;
    
    float ao = 0.0;
    float radius = u_AO_Radius * 0.05;
    int aoSamples = AO_SAMPLES[u_Quality_Mode];
    
    [unroll]
    for(int i = 0; i < aoSamples; i++)
    {
        float angle = (i / float(aoSamples)) * 2.0 * PI;
        float2 offset = float2(cos(angle), sin(angle)) * radius;
        float2 sampleCoord = texcoord + offset;
        
        float sampleDepth = tex2D(samplerDepth, sampleCoord).r;
        float3 samplePos = GetViewPosition(sampleCoord, sampleDepth);
        float3 diffVec = samplePos - GetViewPosition(texcoord, depth);
        
        float weight = saturate(dot(normalize(diffVec), normal));
        ao += weight * (1.0 / (1.0 + length(diffVec)));
    }
    
    ao = 1.0 - (ao / float(aoSamples)) * u_AO_Strength;
    return saturate(ao);
}

float3 DenoiseSSR(float2 texcoord, float3 ssr, float depth, float3 normal)
{
    if(!u_Enable_Denoise) return ssr;
    
    float2 pixelSize = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
    float3 denoisedSSR = 0.0;
    float totalWeight = 0.0;
    
    [unroll]
    for(int x = -1; x <= 1; x++)  // Reduced from -2/2 to -1/1
    {
        [unroll]
        for(int y = -1; y <= 1; y++)  // Reduced from -2/2 to -1/1
        {
            float2 offset = float2(x, y) * pixelSize;
            float2 sampleCoord = texcoord + offset;
            
            float sampleDepth = tex2D(samplerDepth, sampleCoord).r;
            float3 sampleNormal = tex2D(samplerNormal, sampleCoord).rgb * 2.0 - 1.0;
            float3 sampleSSR = tex2D(samplerSSR, sampleCoord).rgb;
            
            float depthWeight = saturate(1.0 - abs(depth - sampleDepth) * 100.0);
            float normalWeight = pow(saturate(dot(normal, sampleNormal)), 8.0);
            float spatialWeight = saturate(1.0 - length(offset) * 4.0);
            
            float weight = depthWeight * normalWeight * spatialWeight;
            weight = pow(weight, u_Denoise_Strength);
            
            denoisedSSR += sampleSSR * weight;
            totalWeight += weight;
        }
    }
    
    return denoisedSSR / max(totalWeight, 0.001);
}

float3 CalculateSSR(float2 texcoord, float3 normal, float depth, float4 material)
{
    if(!u_Enable_SSR) return 0.0;
    
    float reflectivity = material.r * (1.0 - material.g) * u_SSR_Intensity;
    if(reflectivity < 0.01) return 0.0;
    
    float3 viewPos = GetViewPosition(texcoord, depth);
    float3 viewDir = normalize(-viewPos);
    float3 reflectDir = reflect(-viewDir, normal);
    
    float fresnel = GetFresnel(normal, viewDir);
    float edgeFactor = GetEdgeFactor(texcoord, depth, normal);
    
    // Use a fixed step count for the main loop
    const int FIXED_STEPS = 16;
    float3 rayStep = reflectDir * (1.0 / FIXED_STEPS);
    float3 rayPos = viewPos;
    float3 reflection = 0.0;
    float hit = 0.0;
    
    [loop]  // Changed from [unroll] to [loop]
    for(int i = 0; i < FIXED_STEPS; i++)
    {
        rayPos += rayStep;
        
        float4 projPos = mul(float4(rayPos, 1.0), ProjectionMatrix);
        projPos.xy /= projPos.w;
        
        if(any(abs(projPos.xy) > 1.0)) continue;
        
        float2 sampleCoord = projPos.xy * 0.5 + 0.5;
        sampleCoord.y = 1.0 - sampleCoord.y;
        
        if(any(sampleCoord < 0.0) || any(sampleCoord > 1.0)) continue;
        
        float sampleDepth = tex2Dlod(samplerDepth, float4(sampleCoord, 0, 0)).r;
        float3 sampleViewPos = GetViewPosition(sampleCoord, sampleDepth);
        
        float deltaDepth = rayPos.z - sampleViewPos.z;
        if(deltaDepth > 0.0 && deltaDepth < 1.0)
        {
            reflection = tex2Dlod(samplerColor, float4(sampleCoord, 0, 0)).rgb;
            hit = 1.0 - (i / float(FIXED_STEPS));
            break;
        }
    }
    
    return reflection * hit * reflectivity * fresnel * edgeFactor;
}

float3 AdjustSaturation(float3 color, float saturation)
{
    float grey = dot(color, float3(0.2126, 0.7152, 0.0722));
    return lerp(grey.xxx, color, saturation);
}

float3 AdjustContrast(float3 color, float contrast)
{
    float midpoint = 0.5;
    return (color - midpoint) * contrast + midpoint;
}

//================== Vertex Shader ==================
void VS_PostProcess(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD0)
{
    texcoord.x = (id == 2) ? 2.0 : 0.0;
    texcoord.y = (id == 1) ? 2.0 : 0.0;
    position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

//================== Pixel Shaders ==================
void PS_MaterialAnalysis(float4 pos : SV_Position, float2 texcoord : TEXCOORD0, out float4 outMaterial : SV_Target0)
{
    float3 color = tex2D(samplerColor, texcoord).rgb;
    float depth = tex2D(samplerDepth, texcoord).r;
    float3 normal = GetNormals(texcoord);
    
    outMaterial = AnalyzeMaterial(color, normal, depth);
}

void PS_Normals(float4 pos : SV_Position, float2 texcoord : TEXCOORD0, out float4 outNormal : SV_Target0)
{
    outNormal = float4(GetNormals(texcoord) * 0.5 + 0.5, 1.0);
}

void PS_SSS(float4 pos : SV_Position, float2 texcoord : TEXCOORD0, out float4 outSSS : SV_Target0)
{
    if(!u_Enable_SSS)
    {
        outSSS = 0.0;
        return;
    }
    
    float depth = tex2D(samplerDepth, texcoord).r;
    float3 normal = tex2D(samplerNormal, texcoord).rgb * 2.0 - 1.0;
    float4 material = tex2D(samplerMaterial, texcoord);
    
    float3 sssColor = CalculateSSS(texcoord, normal, depth, material.a);
    outSSS = float4(sssColor, 1.0);
}

void PS_SSR(float4 pos : SV_Position, float2 texcoord : TEXCOORD0, out float4 outSSR : SV_Target0)
{
    if(!u_Enable_SSR)
    {
        outSSR = 0.0;
        return;
    }
    
    float depth = tex2D(samplerDepth, texcoord).r;
    float3 normal = tex2D(samplerNormal, texcoord).rgb * 2.0 - 1.0;
    float4 material = tex2D(samplerMaterial, texcoord);
    
    float3 ssr = CalculateSSR(texcoord, normal, depth, material);
    outSSR = float4(ssr, 1.0);
}

void PS_Final(float4 pos : SV_Position, float2 texcoord : TEXCOORD0, out float4 outColor : SV_Target0)
{
    // Sample textures
    float3 color = tex2D(samplerColor, texcoord).rgb;
    float depth = tex2D(samplerDepth, texcoord).r;
    float3 normal = tex2D(samplerNormal, texcoord).rgb * 2.0 - 1.0;
    float3 sssColor = tex2D(samplerSSS, texcoord).rgb;
    float4 material = tex2D(samplerMaterial, texcoord);
    float3 ssr = tex2D(samplerSSR, texcoord).rgb;
    
    // Calculate lighting components
    float3 viewPos = GetViewPosition(texcoord, depth);
    float3 viewDir = normalize(-viewPos);
    float3 lightDir = normalize(float3(1.0, 1.0, -1.0));
    
    // Calculate ambient occlusion
    float ao = CalculateAO(texcoord, normal, depth);
    
    // Denoise SSR if enabled
    if(u_Enable_Denoise)
    {
        ssr = DenoiseSSR(texcoord, ssr, depth, normal);
    }
    
    // Combine all lighting components
    float3 finalColor = color;
    
    // Add SSS only for skin-like materials
    if(u_Enable_SSS)
    {
        finalColor += sssColor * u_SSS_Strength * material.a;
    }
    
    // Add reflections only for non-skin, reflective materials
    if(u_Enable_SSR)
    {
        finalColor += ssr * (1.0 - material.a);
    }
    
    // Apply ambient occlusion
    finalColor *= ao;
    
    // Color grading
    finalColor = AdjustSaturation(finalColor, u_Saturation);
    finalColor = AdjustContrast(finalColor, u_Contrast);
    finalColor *= u_ColorTemp;
    
    // Apply exposure
    finalColor *= u_Exposure;
    
    // Tone mapping (ACES approximation)
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    finalColor = saturate((finalColor * (a * finalColor + b)) / (finalColor * (c * finalColor + d) + e));
    
    // Gamma correction
    finalColor = pow(finalColor, 1.0 / 2.2);
    
    outColor = float4(finalColor, 1.0);
}

//================== Technique ==================
technique VortexRT <
    ui_tooltip = "Advanced rendering with subsurface scattering, reflections and improved lighting";
>
{
    pass MaterialAnalysis
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_MaterialAnalysis;
        RenderTarget0 = texMaterial;
    }
    
    pass GenerateNormals
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_Normals;
        RenderTarget0 = texNormal;
    }
    
    pass SubsurfaceScattering
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_SSS;
        RenderTarget0 = texSSS;
    }
    
    pass ScreenSpaceReflections
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_SSR;
        RenderTarget0 = texSSR;
    }
    
    pass FinalComposition
    {
        VertexShader = VS_PostProcess;
        PixelShader = PS_Final;
    }
}