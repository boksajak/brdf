# Crash Course in BRDF Implementation Code Sample

This is a code sample accompanying the article at https://boksajak.github.io/blog/BRDF. The file can be compiled as a HLSL shader or C++ code.
                                 
# Example

A working path tracer which uses this BRDF implementation can be found in the [ReferencePT project](https://github.com/boksajak/referencePT) on GitHub. See file [PathTracer.hlsl](https://github.com/boksajak/referencePT/blob/master/shaders/PathTracer.hlsl) which includes **brdf.h** and calls to functions **evalCombinedBRDF** and **evalIndirectCombinedBRDF** for details.

# How to use

 1. Include **brdf.h** in your project
 2. If compiling as C++, make sure that supporting [GLM library](https://github.com/g-truc/glm) is available and its includes at the beginning of the **brdf.h** file are valid. This library makes it possible to use HLSL-style structures and functions and compile it as C++ with minimal effort.
 3. Call functions **evalCombinedBRDF** and **evalIndirectCombinedBRDF** to evaluate BRDFs as described below and in the example. When compiling for C++, all functions and structures are placed in the **brdf** namespace.
 
## Direct light evaluation

To evaluate BRDF for light coming from a light source, use the function **evalCombinedBRDF**, as follows:

```cpp
const float3 V = -ray.Direction;            //< V is the direction towards the viewer
const float3 L = normalize(lightVector);    //< L is the direction towards the light source
const float3 N = shadingNormal;             //< N is the shading normal

MaterialProperties material = ...           //< Initialize material properties to that of evaluated surface 
material.reflectance = 0.5f;                //< If reflectance of the surface is used (see below) but not specified, 
                                            //< it should be initialized to 0.5 

float3 light = evalCombinedBRDF(N, L, V, material);
```

## Indirect light evaluation

To generate a reflected ray according to the BRDF of the surface (and essentially evaluate indirect light contribution to the surface), use the function **evalIndirectCombinedBRDF** as follows:

```cpp
const float2 random = ...        //< 2 random numbers in the interval <0,1). Note that interval is open on the right 
                                 //< side - number 1 can never occur here as it would generate NaNs when sampling the BRDF
const int brdfType = ...         //< Choose between SPECULAR_TYPE and DIFFUSE_TYPE. An example of how to choose in a path tracer 
                                 //< is in [ReferencePT project](https://github.com/boksajak/referencePT)

float3 rayDirection;             //< Sampled ray direction will be here
float3 brdfWeight;               //< The weight of the BRDF sample will be here

if (!evalIndirectCombinedBRDF(random, shadingNormal, geometryNormal, V, material, brdfType, rayDirection, brdfWeight)) 
{
    break; // Failed to generate valid ray direction
}
    
```

# Options

The **brdf.h** file is configurable by macros at its beginning.
 
 
## Minimal reflectance specification

By default, the specular reflectance of dielectrics (materials with metalness set to zero) is specified by definition **MIN_DIELECTRICS_F0** to 4%. 

If you want to let users to specify dielectrics reflectance per material, you can enable macro **USE_REFLECTANCE_PARAMETER** and set the __reflectance__ parameter of the **MaterialProperties** structure. Note that actual reflectance is calculated as __0.16 * reflectance * reflectance__. This makes changes to the parameter to appear more linear. If reflectance parameter is used, but some material doesn't specify it, it should be initialized to 0.5 - this will result in the reflectance of 4% as per formula above. 
 
## VNDF Sampling

By default, the implementation uses VNDF sampling from __"Sampling the GGX Distribution of Visible Normals"__ by Heitz. If you want to use newer sampling from __"Sampling Visible GGX Normals with Spherical Caps"__ by Dupuy & Benyoub, enable macro **USE_VNDF_WITH_SPHERICAL_CAPS**. This new method can bring performance benefits.

There is also an option to use older Walter's sampling from __"Microfacet Models for Refraction through Rough Surfaces"__ by enablinf macro **USE_WALTER_GGX_SAMPLING**.  
 
## FUNCTION macro

All functions are prefixed with the macro called **FUNCTION**, which is set to **static** keyword when compiling as C++, and is empty in HLSL.

## Release notes

### brdf.h 1.2 - August 2023
#### Features:
- Added VNDF sampling from __"Sampling Visible GGX Normals with Spherical Caps"__ by Dupuy & Benyoub
- Added Walter's sampling of GG-X distribution
- Added optional reflectance parameter to specify minimal reflectance of dielectrics per material
- Added function **specularGGXReflectanceApprox** from "Accurate Real-Time Specular Reflections with Radiance Caching" in Ray Tracing Gems by Hirvonen et al.
- All functions and structures put into **brdf** namespace when compiling as C++
- All functions specified as static when compiling as C++
