
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
//#include <optix_world.h>


using namespace optix;

// This is to be plugged into an RTgeometryInstance object to represent
// a triangle mesh from mitsuba with a vertex info buffer containing
// vertex position, normal (optional), texture coordinates (optional), 
// UV Tangents (optional) and color (optional).

rtBuffer<float> vertex_info_buffer;
rtBuffer<uint3> index_buffer;

rtDeclareVariable(int, stride, , );
rtDeclareVariable(int, normal_offset, , );
rtDeclareVariable(int, texCoord_offset, , );
rtDeclareVariable(int, UVTangent_offset, , );
rtDeclareVariable(int, color_offset, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


RT_PROGRAM void mesh_intersect(int primIdx)
{	
	uint3 v_idx = index_buffer[primIdx];

	float3 p0 = make_float3(vertex_info_buffer[v_idx.x*stride], vertex_info_buffer[v_idx.x*stride+1], vertex_info_buffer[v_idx.x*stride+2]);
	float3 p1 = make_float3(vertex_info_buffer[v_idx.y*stride], vertex_info_buffer[v_idx.y*stride+1], vertex_info_buffer[v_idx.y*stride+2]);
	float3 p2 = make_float3(vertex_info_buffer[v_idx.z*stride], vertex_info_buffer[v_idx.z*stride+1], vertex_info_buffer[v_idx.z*stride+2]);

	// Intersect ray with triangle
	float3 n;
	float  t, beta, gamma;
	if (intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma)) {

		if (rtPotentialIntersection(t)) {


			if (normal_offset==0) {
				shading_normal = normalize(n);
				//shading_normal = make_float3(1.0,0.0,0.0);
			} else {
				float3 n0 = make_float3(vertex_info_buffer[v_idx.x*stride + normal_offset],
										vertex_info_buffer[v_idx.x*stride + normal_offset + 1],
										vertex_info_buffer[v_idx.x*stride + normal_offset + 2]);
				float3 n1 = make_float3(vertex_info_buffer[v_idx.y*stride + normal_offset],
										vertex_info_buffer[v_idx.y*stride + normal_offset + 1],
										vertex_info_buffer[v_idx.y*stride + normal_offset + 2]);
				float3 n2 = make_float3(vertex_info_buffer[v_idx.z*stride + normal_offset],
										vertex_info_buffer[v_idx.z*stride + normal_offset + 1],
										vertex_info_buffer[v_idx.z*stride + normal_offset + 2]);
				shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f - beta - gamma));
				//shading_normal = make_float3(0.0, 1.0, 0.0);
			}
			geometric_normal = normalize(n);

			//int3 t_idx = tindex_buffer[primIdx];
			//if (texcoord_buffer.size() == 0 || t_idx.x < 0 || t_idx.y < 0 || t_idx.z < 0) {
			//	texcoord = make_float3(0.0f, 0.0f, 0.0f);
			//} else {
			//	float2 t0 = texcoord_buffer[t_idx.x];
			//	float2 t1 = texcoord_buffer[t_idx.y];
			//	float2 t2 = texcoord_buffer[t_idx.z];
			//	texcoord = make_float3(t1*beta + t2*gamma + t0*(1.0f - beta - gamma));
			//}

			rtReportIntersection(0);
		}
	}
}

RT_PROGRAM void mesh_bounds(int primIdx, float result[6])
{
	const uint3 v_idx = index_buffer[primIdx];

	const float3 v0 = make_float3(vertex_info_buffer[v_idx.x*stride], vertex_info_buffer[v_idx.x*stride + 1], vertex_info_buffer[v_idx.x*stride + 2]);
	const float3 v1 = make_float3(vertex_info_buffer[v_idx.y*stride], vertex_info_buffer[v_idx.y*stride + 1], vertex_info_buffer[v_idx.y*stride + 2]);
	const float3 v2 = make_float3(vertex_info_buffer[v_idx.z*stride], vertex_info_buffer[v_idx.z*stride + 1], vertex_info_buffer[v_idx.z*stride + 2]);

	const float  area = length(cross(v1 - v0, v2 - v0));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if (area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf(fminf(v0, v1), v2);
		aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
	} else {
		aabb->invalidate();
	}
}

