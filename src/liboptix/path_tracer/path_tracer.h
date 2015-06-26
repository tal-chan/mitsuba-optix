
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */
#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <optixu/optixu_math_namespace.h>
#include <optix_host.h>
#include <SampleScene.h>


struct ParallelogramLight
{
  optix::float3 corner;
  optix::float3 v1, v2;
  optix::float3 normal;
  optix::float3 emission;
  bool textured;
};

static __device__ __inline__ void mapToDisk( optix::float2& sample )
{
  float phi, r;
  float a = 2.0f * sample.x - 1.0f;      // (a,b) is now on [-1,1]^2 
  float b = 2.0f * sample.y - 1.0f;      // 
  if (a > -b) {                           // reg 1 or 2 
    if (a > b) {                          // reg 1, also |a| > |b| 
      r = a;
      phi = (M_PIf/4.0f) * (b/a);
    } else {                              // reg 2, also |b| > |a| 
      r = b;
      phi = (M_PIf/4.0f) * (2.0f - a/b);
    }
  } else {                                // reg 3 or 4 
    if (a < b) {                          // reg 3, also |a|>=|b| && a!=0 
      r = -a;
      phi = (M_PIf/4.0f) * (4.0f + b/a);
    } else {                              // region 4, |b| >= |a|,  but 
      // a==0 and  b==0 could occur. 
      r = -b;
      phi = b != 0.0f ? (M_PIf/4.0f) * (6.0f - a/b) :
        0.0f;
    }
  }
  float u = r * cosf( phi );
  float v = r * sinf( phi );
  sample.x = u;
  sample.y = v;
}


// Uniformly sample the surface of a Parallelogram.  Return probability
// of the given sample
static
__device__ __inline__ void sampleParallelogram( const ParallelogramLight& light, 
                                                const optix::float3& hit_point,
                                                const optix::float2& sample,
                                                optix::float3& w_in,
                                                float& dist,
                                                float& pdf )
{
  using namespace optix;

  float3 on_light = light.corner + sample.x*light.v1 + sample.y*light.v2;
  w_in = on_light - hit_point;
  float dist2 = dot( w_in, w_in );
  dist  = sqrt( dist2 ); 
  w_in /= dist;
  
  float3 normal = cross( light.v1, light.v2 );
  float area    = length( normal );
  normal /= area;
  float cosine  = -dot( normal, w_in );
  pdf = dist2 / ( area * cosine );
}


// Create ONB from normal.  Resulting W is Parallel to normal
static
__device__ __inline__ void createONB( const optix::float3& n,
                                      optix::float3& U,
                                      optix::float3& V,
                                      optix::float3& W )
{
  using namespace optix;

  W = normalize( n );
  U = cross( W, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( fabsf( U.x) < 0.001f && fabsf( U.y ) < 0.001f && fabsf( U.z ) < 0.001f  )
    U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( W, U );
}

// Create ONB from normalalized vector
static
__device__ __inline__ void createONB( const optix::float3& n,
                                      optix::float3& U,
                                      optix::float3& V)
{
  using namespace optix;

  U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( dot(U, U) < 1.e-3f )
    U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( n, U );
}

// sample hemisphere with cosine density
static
__device__ __inline__ void sampleUnitHemisphere( const optix::float2& sample,
                                                 const optix::float3& U,
                                                 const optix::float3& V,
                                                 const optix::float3& W,
                                                 optix::float3& point )
{
    using namespace optix;

    float phi = 2.0f * M_PIf*sample.x;
    float r = sqrt( sample.y );
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = 1.0f - x*x -y*y;
    z = z > 0.0f ? sqrt(z) : 0.0f;

    point = x*U + y*V + z*W;
}

class PathTracerScene : public SampleScene
{
public:
	// Set the actual render parameters below in main().
	PathTracerScene()
		: m_rr_begin_depth(1u)
		, m_sqrt_num_samples(0u)
		, m_width(512u)
		, m_height(512u)
	{}

	virtual void   initScene(InitialCameraData& camera_data);
	virtual void   trace(const RayGenCameraData& camera_data);
	virtual optix::Buffer getOutputBuffer();

	void finishGeometry();
	void validateAndCompile();
	void   setNumSamples(unsigned int sns)                           { m_sqrt_num_samples = sns; }
	void   setDimensions(const unsigned int w, const unsigned int h) { m_width = w; m_height = h; }
	void   renderToActiveTexture();
	void addTransformedShape(unsigned int id[2],
		unsigned int triCount,
		unsigned int vertexCount,
		bool hasNormals,
		bool hasTexCoord,
		bool hasUVTangents,
		bool hasColor,
		float transform[16]);

private:
	// Should return true if key was handled, false otherwise.
	virtual bool keyPressed(unsigned char key, int x, int y);
	void createGeometry();

	optix::GeometryInstance createParallelogram(const float3& anchor,
		const float3& offset1,
		const float3& offset2);

	optix::GeometryInstance createLightParallelogram(const float3& anchor,
		const float3& offset1,
		const float3& offset2,
		int lgt_instance = -1);
	void setMaterial(optix::GeometryInstance& gi,
		optix::Material material,
		const std::string& color_name,
		const float3& color);

	optix::Material m_diffuse;
	optix::Group m_top_object;
	optix::Group m_shadow_object;

	optix::Program        m_pgram_bounding_box;
	optix::Program        m_pgram_intersection;

	optix::Program        m_pgram_trimesh_bounding_box;
	optix::Program        m_pgram_trimesh_intersection;

	unsigned int   m_rr_begin_depth;
	unsigned int   m_sqrt_num_samples;
	unsigned int   m_width;
	unsigned int   m_height;
	unsigned int   m_frame;
	unsigned int   m_sampling_strategy;
};

#endif