/*
    This file is part of Dirt, the Dartmouth introductory ray tracer.

    Copyright (c) 2017-2019 by Wojciech Jarosz

    Dirt is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Dirt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <dirt/scene.h>
#include <dirt/progress.h>
#include <dirt/sampler.h>
#include <fstream>
#include "mpi.h"
#include <cmath> // sqrt

/// Construct a new scene from a json object
Scene::Scene(const json & j)
{
    parseFromJSON(j);
}

/// Read a scene from a json file
Scene::Scene(const string & filename)
{
    // open file
    std::ifstream stream(filename, std::ifstream::in);
    if (!stream.good())
    	throw DirtException("Cannot open file: %s.", filename);

    json j;
    stream >> j;
    parseFromJSON(j);
}

Scene::~Scene()
{
    m_materials.clear();
}


shared_ptr<const Material> Scene::findOrCreateMaterial(const json & jp, const string& key) const
{
    auto it = jp.find(key);
    if (it == jp.end())
        return Material::defaultMaterial();
    
    auto j = it.value();
    if (j.is_string())
    {
        string name = j.get<string>();
        // find a pre-declared material
        auto i = m_materials.find(name);
        if (i != m_materials.end())
            return i->second;
        else
            throw DirtException("Can't find a material with name '%s' here:\n%s", name, jp.dump(4));
    }
    else if (j.is_object())
    {
	    // create a new material
        return parseMaterial(j);
    }
    else
        throw DirtException("Type mismatch: Expecting either a material or material name here:\n%s", jp.dump(4));
}

shared_ptr<const Medium> Scene::findOrCreateMedium(const json & jp, const string& key) const
{
    auto it = jp.find(key);
    if (it == jp.end())
        return nullptr;

    auto j = it.value();
    if (j.is_string())
    {
        string name = j.get<string>();
        // find a pre-declared medium
        auto i = m_media.find(name);
        if (i != m_media.end())
            return i->second;
        else
            throw DirtException("Can't find a medium with name '%s' here:\n%s", name, jp.dump(4));
    }
    else if (j.is_object())
    {
	    // create a new medium
        return parseMedium(j);
    }
    else
        throw DirtException("Type mismatch: Expecting either a medium or medium name here:\n%s", jp.dump(4));
}

shared_ptr<const MediumInterface> Scene::findOrCreateMediumInterface(const json & jp, const string& key) const
{
    auto it = jp.find(key);
    if (it == jp.end())
    {
        return std::make_shared<MediumInterface>();
    }
    std::shared_ptr<const Medium> inside = findOrCreateMedium(jp.at(key), "inside");
    std::shared_ptr<const Medium> outside = findOrCreateMedium(jp.at(key), "outside");
    return std::make_shared<MediumInterface>(inside, outside);
}

// compute the color corresponding to a ray by raytracing
Color3f Scene::recursiveColor(Sampler &sampler, const Ray3f &ray, int depth) const
{
    // Pseudo-code:
    //
	// if scene.intersect:
    //      get emitted color (hint: you can use hit.mat->emitted)
	// 		if depth < MaxDepth and hit_material.scatter(....) is successful:
	//			recursive_color = call this function recursively with the scattered ray and increased depth
	//          return emitted color + attenuation * recursive_color
	//		else
	//			return emitted color;
	// else:
	// 		return background color (hint: look at m_background)
    const int maxDepth = 64;
    HitInfo hit;
    if (intersect(ray, hit))
    {
        Ray3f scattered;
        Color3f attenuation;
        Color3f emitted = hit.mat->emitted(ray, hit);
        Vec2f sample = sampler.next2D();
        if (depth < maxDepth && hit.mat->scatter(ray, hit, sample, attenuation, scattered))
        {
            return emitted + attenuation * recursiveColor(sampler, scattered, depth + 1);
        }
        else
        {
            return emitted;
        }
    }
    else
    {
        return m_background;
    }
}

// raytrace an image
Image3f Scene::raytrace() const
{
	std::cout << "RAYTRACE" << " " << m_surfaces->localBBox().pMin << std::endl;
    std::cout << "RAYTRACE" << " " << m_surfaces->localBBox().pMax << std::endl;
    int pid;
    int nproc;

    // Initialize MPI
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes specificed at start of run
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // New MPI datatype
    MPI_Datatype pixelType;
    int blockLens[1];
    MPI_Aint offsets[1];
    MPI_Datatype oldTypes[1];

    blockLens[0] = 3;

    // MPI_Aint extent, lb;
    // MPI_Type_get_extent(MPI_FLOAT, &lb, &extent);
    // offsets[0] = 0;
    offsets[0] = 0;
    // offsets[1] = extent;

    oldTypes[0] = MPI_FLOAT;
    MPI_Type_create_struct(1, blockLens, offsets, oldTypes, &pixelType);
    MPI_Type_commit(&pixelType);

    // allocate an image of the proper size
    auto image = Image3f(m_camera->resolution().x, m_camera->resolution().y);

    if (m_integrator)
        return integrateImage();

    // Pseudo-code:
    //
        // foreach image row (go over image height)
            // foreach pixel in the row (go over image width)
                // init accumulated color to zero
                // repeat m_imageSamples times:
                    // compute a random point within the pixel (you can just add a random number between 0 and 1
                    //                                          to the pixel coordinate. You can use randf() for this)
                    // compute camera ray
                    // accumulate color raytraced with the ray (by calling recursiveColor)
                // divide color by the number of pixel samples

    // Hint: you can create a Progress object (progress.h) to provide a 
    // progress bar during rendering.

    Progress progress("Rendering", m_camera->resolution().x*m_camera->resolution().y);

    // no MPI
    // int ind_i_min = 0;
    // int ind_i_max = m_camera->resolution().x;
    // int ind_j_min = 0;
    // int ind_j_max = m_camera->resolution().y;

    // Grid based parallelism
    // Divide up the space similar to nbody simulation
    // Need to figure out granularity of communication

    // foreach pixel
    // Divide up into grid, compute own section, gather back to thread 0

    // Divide up pixels by scattering
    // Square Grid
    // int squareSize = static_cast<int>(pow(nproc, 0.5));
    // int row_num = pid / squareSize;
    // int col_num = pid % squareSize;
    // int i_len = m_camera->resolution().x / squareSize;
    // int j_len = m_camera->resolution().y / squareSize;
    // int ind_i_min = row_num * i_len;
    // int ind_i_max = (row_num+1) * i_len;
    // int ind_j_min = col_num * j_len;
    // int ind_j_max = (col_num+1) * j_len;
    // std::cout << ">>> " << pid << " " << ind_i_min << " " << ind_i_max << " " << ind_j_min << " " << ind_j_max << "\n";
    
    // Rows
    // int row_num = m_camera->resolution().y / nproc;
    // int ind_i_min = 0;
    // int ind_i_max = m_camera->resolution().x;
    // int ind_j_min = pid*row_num;
    // int ind_j_max = (pid+1)*row_num;

    // Columns
    // int col_num = m_camera->resolution().x / nproc;
    // int ind_i_min = pid*col_num;
    // int ind_i_max = (pid+1)*col_num;
    // int ind_j_min = 0;
    // int ind_j_max = m_camera->resolution().y;

    // Load balance rows
    int concentration = ceil(m_surfaces->localBBox().pMax[1] - m_surfaces->localBBox().pMin[1]);
    int offset, row_num, ind_i_min, ind_i_max, ind_j_min, ind_j_max;
    if (concentration >= m_camera->resolution().y-1) {
        row_num = m_camera->resolution().y / nproc;
        ind_i_min = 0;
        ind_i_max = m_camera->resolution().x;
        ind_j_min = pid*row_num;
        ind_j_max = (pid+1)*row_num;
    } else {
        if (concentration % 2 == 1) concentration++;
        offset = (m_camera->resolution().y - concentration) / 2;
        row_num = concentration / nproc;
        ind_i_min = 0;
        ind_i_max = m_camera->resolution().x;
        ind_j_min = pid*row_num+offset;
        ind_j_max = (pid+1)*row_num+offset;
    }
    
    // Need to send data, reconstruct image class
    // MPI_Scatter(&newParticles[leftOver], chunkSize, particleType, &particles[0], chunkSize, particleType, 0, MPI_COMM_WORLD);
    std::cout << "<<< " << pid << " " << ind_i_min << " " << ind_i_max << " " << ind_j_min << " " << ind_j_max << "\n";


	// #pragma omp parallel
    for (int j=ind_j_min; j<ind_j_max; j++)
    {
        for (int i=ind_i_min; i<ind_i_max; i++)
        {
            // init accumulated color
            image(i, j) = Color3f(0.f);

            m_sampler->startPixel();

            // foreach sample
            for (int s = 0; s < m_imageSamples; ++s)
            {
                // set pixel to the color raytraced with the ray
                INCREMENT_TRACED_RAYS;
                Vec2f sample = m_sampler->next2D();
                image(i, j) += recursiveColor(*m_sampler, m_camera->generateRay(i + sample.x, j + sample.y), 0);
                m_sampler->startNextPixelSample();
            }
            // scale by the number of samples
            image(i, j) /= float(m_imageSamples);

            ++progress;
        }
    }
    // MPI_Gatherv(&particles[0], sendCount, particleType, &newParticles[0], &counts[0], &displacements[0], particleType, 0, MPI_COMM_WORLD);
    // MPI_Allgather(&image.m_data[0], i_len * j_len, pixelType, &image.m_data[0], i_len * j_len, pixelType, MPI_COMM_WORLD);
    
    // rows
    // MPI_Allgather(&image.m_data[ind_j_min*m_camera->resolution().x], row_num*m_camera->resolution().x, pixelType, &image.m_data[0], row_num*m_camera->resolution().x, pixelType, MPI_COMM_WORLD);
    
    // cols
    // MPI_Allgather(&image.m_data[ind_j_min*m_camera->resolution().y], col_num*m_camera->resolution().y, pixelType, &image.m_data[0], col_num*m_camera->resolution().y, pixelType, MPI_COMM_WORLD);
    
    // grid (reordering after gathering)
    // auto buffer = Image3f(m_camera->resolution().x, m_camera->resolution().y);
    // for (int i=0; i<squareSize; i++) {
    //     MPI_Allgather(&image.m_data[ind_i_min + i * m_camera->resolution().x], squareSize, pixelType, &image.m_data[nproc * squareSize * i], squareSize, pixelType, MPI_COMM_WORLD);
    // }
    // for (int i = 0; i < m_camera->resolution().x; i++) {
    //     for (int j = 0; j < m_camera->resolution().y; j++) {
    //         image(i,j) = buffer(); 
    //     }
    // }
    // int l = 0;
    // for (int i = 0; i < squareSize; i++) {
    //     for (int j = 0; j < squareSize; j++) {
    //         for (int k = 0; k < squareSize * squareSize; k++) {
    //             image.m_data[l] = buffer.m_data[j * squareSize * squareSize * squareSize + i * squareSize * squareSize + k];
    //             l++;
    //         }
    //     }
    // }
    // for (int i = 0; i < squareSize; i++) {
    //     for (int j = 0; j < squareSize * squareSize; j++) {
    //         image.m_data[i*squareSize + j] = buffer.m_data[i*squareSize*squareSize*squareSize+j];
    //     }
    // }
    
    // load balance rows
    MPI_Allgather(&image.m_data[ind_j_min*m_camera->resolution().x], row_num*m_camera->resolution().x, pixelType, &image.m_data[0], row_num*m_camera->resolution().x, pixelType, MPI_COMM_WORLD);
    std::vector<int> counts, displacements;
    for (int j=0; j<offset; j++)
    {
        for (int i=ind_i_min; i<ind_i_max; i++)
        {
            // init accumulated color
            image(i, j) = Color3f(0.f);

            m_sampler->startPixel();

            // foreach sample
            for (int s = 0; s < m_imageSamples; ++s)
            {
                // set pixel to the color raytraced with the ray
                INCREMENT_TRACED_RAYS;
                Vec2f sample = m_sampler->next2D();
                image(i, j) += recursiveColor(*m_sampler, m_camera->generateRay(i + sample.x, j + sample.y), 0);
                m_sampler->startNextPixelSample();
            }
            // scale by the number of samples
            image(i, j) /= float(m_imageSamples);

            ++progress;
        }
    }
    for (int j=m_camera->resolution().y-offset; j<m_camera->resolution().y;j++)
    {
        for (int i=ind_i_min; i<ind_i_max; i++)
        {
            // init accumulated color
            image(i, j) = Color3f(0.f);

            m_sampler->startPixel();

            // foreach sample
            for (int s = 0; s < m_imageSamples; ++s)
            {
                // set pixel to the color raytraced with the ray
                INCREMENT_TRACED_RAYS;
                Vec2f sample = m_sampler->next2D();
                image(i, j) += recursiveColor(*m_sampler, m_camera->generateRay(i + sample.x, j + sample.y), 0);
                m_sampler->startNextPixelSample();
            }
            // scale by the number of samples
            image(i, j) /= float(m_imageSamples);

            ++progress;
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);

	// return the ray-traced image
    // return image;

    // Need to figure out how to distinguish who to trust, can avoid with allgather
    return image;
}

Image3f Scene::integrateImage() const
{
	std::cout << "INTEGRATE" << std::endl;
    // allocate an image of the proper size
    auto image = Image3f(m_camera->resolution().x, m_camera->resolution().y);

    Progress progress("Rendering", m_camera->resolution().x*m_camera->resolution().y);
    int pid;
    int nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes specificed at start of run
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // New MPI datatype
    MPI_Datatype pixelType;
    int blockLens[1];
    MPI_Aint offsets[1];
    MPI_Datatype oldTypes[1];

    blockLens[0] = 3;

    // MPI_Aint extent, lb;
    // MPI_Type_get_extent(MPI_FLOAT, &lb, &extent);
    // offsets[0] = 0;
    offsets[0] = 0;
    // offsets[1] = extent;

    oldTypes[0] = MPI_FLOAT;
    MPI_Type_create_struct(1, blockLens, offsets, oldTypes, &pixelType);
    MPI_Type_commit(&pixelType);

    // grid
    // int squareSize = static_cast<int>(pow(nproc, 0.5));
    // int row_num = pid / squareSize;
    // int col_num = pid % squareSize;
    // int i_len = m_camera->resolution().x / squareSize;
    // int j_len = m_camera->resolution().y / squareSize;
    // int ind_i_min = row_num * i_len;
    // int ind_i_max = (row_num+1) * i_len;
    // int ind_j_min = col_num * j_len;
    // int ind_j_max = (col_num+1) * j_len;

    // Rows
    // int row_num = m_camera->resolution().y / nproc;
    // int ind_i_min = 0;
    // int ind_i_max = m_camera->resolution().x;
    // int ind_j_min = pid*row_num;
    // int ind_j_max = (pid+1)*row_num;

    // Columns
    // int col_num = m_camera->resolution().x / nproc;
    // int ind_i_min = pid*col_num;
    // int ind_i_max = (pid+1)*col_num;
    // int ind_j_min = 0;
    // int ind_j_max = m_camera->resolution().y;

    // Load balance rows
    int concentration = ceil(m_surfaces->localBBox().pMax[1] - m_surfaces->localBBox().pMin[1]);
    int offset, row_num, ind_i_min, ind_i_max, ind_j_min, ind_j_max;
    if (concentration >= m_camera->resolution().y-1) {
        row_num = m_camera->resolution().y / nproc;
        ind_i_min = 0;
        ind_i_max = m_camera->resolution().x;
        ind_j_min = pid*row_num;
        ind_j_max = (pid+1)*row_num;
    } else {
        if (concentration % 2 == 1) concentration++;
        offset = (m_camera->resolution().y - concentration) / 2;
        row_num = concentration / nproc;
        ind_i_min = 0;
        ind_i_max = m_camera->resolution().x;
        ind_j_min = pid*row_num+offset;
        ind_j_max = (pid+1)*row_num+offset;
    }

    std::cout << "<<< " << pid << " " << ind_i_min << " " << ind_i_max << " " << ind_j_min << " " << ind_j_max << "\n";

    // foreach pixel
	// #pragma omp for
    for (int j=ind_j_min; j<ind_j_max; j++)
    {
        for (int i=ind_i_min; i<ind_i_max; i++)
        {
            // init accumulated color
            image(i, j) = Color3f(0.f);
            
            m_sampler->startPixel();

            // foreach sample
            for (int s = 0; s < m_imageSamples; ++s)
            {
                // set pixel to the color raytraced with the ray
                INCREMENT_TRACED_RAYS;
                Vec2f sample = m_sampler->next2D();
                Color3f c = m_integrator->Li(*this, *m_sampler, m_camera->generateRay(i + sample.x, j + sample.y));
                // if(c.r != c.b)
                // cout << c << endl;
				// #pragma omp critical
                image(i, j) += c;

				// #pragma omp critical
                m_sampler->startNextPixelSample();
            }
            // scale by the number of samples
			// #pragma omp critical
            image(i, j) /= m_imageSamples;

			// #pragma omp critical
            ++progress;
        }
    }
    // rows
    MPI_Allgather(&image.m_data[ind_j_min*m_camera->resolution().x], row_num*m_camera->resolution().x, pixelType, &image.m_data[0], row_num*m_camera->resolution().x, pixelType, MPI_COMM_WORLD);
    // cols
    // MPI_Allgather(&image.m_data[ind_j_min*m_camera->resolution().y], col_num*m_camera->resolution().y, pixelType, &image.m_data[0], col_num*m_camera->resolution().y, pixelType, MPI_COMM_WORLD);

    // load balance rows
    MPI_Allgather(&image.m_data[ind_j_min*m_camera->resolution().x], row_num*m_camera->resolution().x, pixelType, &image.m_data[0], row_num*m_camera->resolution().x, pixelType, MPI_COMM_WORLD);
    std::vector<int> counts, displacements;
    for (int j=0; j<offset; j++)
    {
        for (int i=ind_i_min; i<ind_i_max; i++)
        {
            // init accumulated color
            image(i, j) = Color3f(0.f);

            m_sampler->startPixel();

            // foreach sample
            for (int s = 0; s < m_imageSamples; ++s)
            {
                // set pixel to the color raytraced with the ray
                INCREMENT_TRACED_RAYS;
                Vec2f sample = m_sampler->next2D();
                image(i, j) += recursiveColor(*m_sampler, m_camera->generateRay(i + sample.x, j + sample.y), 0);
                m_sampler->startNextPixelSample();
            }
            // scale by the number of samples
            image(i, j) /= float(m_imageSamples);

            ++progress;
        }
    }
    for (int j=m_camera->resolution().y-offset; j<m_camera->resolution().y;j++)
    {
        for (int i=ind_i_min; i<ind_i_max; i++)
        {
            // init accumulated color
            image(i, j) = Color3f(0.f);

            m_sampler->startPixel();

            // foreach sample
            for (int s = 0; s < m_imageSamples; ++s)
            {
                // set pixel to the color raytraced with the ray
                INCREMENT_TRACED_RAYS;
                Vec2f sample = m_sampler->next2D();
                image(i, j) += recursiveColor(*m_sampler, m_camera->generateRay(i + sample.x, j + sample.y), 0);
                m_sampler->startNextPixelSample();
            }
            // scale by the number of samples
            image(i, j) /= float(m_imageSamples);

            ++progress;
        }
    }


	// return the ray-traced image
    return image;
}
