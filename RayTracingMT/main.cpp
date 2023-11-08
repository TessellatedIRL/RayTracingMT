
#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "random.h"
#include "sphere.h"

#include <float.h>
#include <iostream>

#include <fstream>
#include <chrono>
#include <string>

#include <thread>
#include <vector>
#include <mutex>
#include <atomic>

vec3 color(const ray& r, hittable* world, int depth) 
{
	hit_record rec;
	if (world->hit(r, 0.001, FLT_MAX, rec)) 
	{
		ray scattered;
		vec3 attenuation;
		if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) 
		{
			return attenuation * color(scattered, world, depth + 1);
		}
		else 
		{
			return vec3(0, 0, 0);
		}
	}
	else 
	{
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}


hittable* random_scene() 
{
	int n = 500;
	hittable** list = new hittable * [n + 1];
	list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
	int i = 1;
	for (int a = -11; a < 11; a++) 
	{
		for (int b = -11; b < 11; b++) 
		{
			float choose_mat = random_double();
			vec3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
			if ((center - vec3(4, 0.2, 0)).length() > 0.9) 
			{
				if (choose_mat < 0.8) 
				{  // diffuse
					list[i++] = new sphere(center, 0.2,
						new lambertian(vec3(random_double() * random_double(), random_double() * random_double(), random_double() * random_double())));
				}
				else if (choose_mat < 0.95) 
				{ // metal
					list[i++] = new sphere(center, 0.2,
						new metal(vec3(0.5 * (1 + random_double()), 0.5 * (1 + random_double()), 0.5 * (1 + random_double())), 0.5 * random_double()));
				}
				else 
				{  // glass
					list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
	}

	list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
	list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
	list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

	return new hittable_list(list, i);
}

std::mutex writeM;

struct BlockJob
{
	int rowStart;
	int rowEnd;
	int colSize;
	int spp;
	std::vector<int> indices;
	std::vector<vec3> colors;
};

void CalculateColor(BlockJob job, std::vector<BlockJob>& imageBlocks, int ny, camera cam, hittable* world,
	std::mutex& mutex, std::condition_variable& cv, std::atomic<int>& completedThreads)
{
	for (int j = job.rowStart; j < job.rowEnd; ++j) 
	{
		for (int i = 0; i < job.colSize; ++i) 
		{
			vec3 col(0, 0, 0);
			for (int s = 0; s < job.spp; ++s) 
			{
				float u = float(i + random_double()) / float(job.colSize);
				float v = float(j + random_double()) / float(ny);
				ray r = cam.get_ray(u, v);
				col += color(r, world, 0);
			}
			col /= float(job.spp);
			col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

			const unsigned int index = j * job.colSize + i;
			job.indices.push_back(index);
			job.colors.push_back(col);
		}
	}

	{
		std::lock_guard<std::mutex> lock(mutex);
		imageBlocks.push_back(job);
		completedThreads++;
		cv.notify_one();
	}
}

int main() 
{
	int nx = 1200;
	int ny = 800;
	int ns = 10;
	int pixelCount = nx * ny;
	// std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	hittable* world = random_scene();

	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	float dist_to_focus = 10.0f;
	float aperture = 0.1f;

	vec3* image = new vec3[pixelCount];
	memset(&image[0], 0, pixelCount * sizeof(vec3));

	camera cam(lookfrom, lookat, vec3(0, -1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus);

	auto fulltime = std::chrono::high_resolution_clock::now();

	const int nThreads = std::thread::hardware_concurrency();
	int rowsPerThread = ny / nThreads;
	int leftOver = ny % nThreads;

	std::mutex mutex;
	std::condition_variable cvResults;
	std::vector<BlockJob> imageBlocks;
	std::atomic<int> completedThreads = { 0 };
	std::vector<std::thread> threads;

	for (int i = 0; i < nThreads; ++i)
	{
		BlockJob job;
		job.rowStart = i * rowsPerThread;
		job.rowEnd = job.rowStart + rowsPerThread;
		if (i == nThreads - 1)
		{
			job.rowEnd = job.rowStart + rowsPerThread + leftOver;
		}
		job.colSize = nx;
		job.spp = ns;

		std::thread t([job, &imageBlocks, ny, &cam, &world, &mutex, &cvResults, &completedThreads]() 
			{
				CalculateColor(job, imageBlocks, ny, cam, world, mutex, cvResults, completedThreads);
			});
		threads.push_back(std::move(t));
	}

	// Launched jobs
	// Need to build image.
	// Wait for number of jobs = pixel count
	{
		std::unique_lock<std::mutex> lock(mutex);
		cvResults.wait(lock, [&completedThreads, &nThreads] 
			{
				return completedThreads == nThreads;
			});
	}

	for (std::thread& t : threads)
	{
		t.join();
	}

	for (BlockJob job : imageBlocks)
	{
		int index = job.rowStart;
		int colorIndex = 0;
		for (vec3& col : job.colors)
		{
			int colIndex = job.indices[colorIndex];
			image[colIndex] = col;
			++colorIndex;
		}
	}

	auto timeSpan = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - fulltime);
	int frameTimeMs = static_cast<int>(timeSpan.count());
	std::cout << " - time " << frameTimeMs << " ms \n";

	std::string filename = "Multithreaded.ppm";

	std::ofstream fileHandler;
	fileHandler.open(filename, std::ios::out | std::ios::binary);
	if (!fileHandler.is_open())
	{
		return false;
	}

	fileHandler << "P3\n" << nx << " " << ny << "\n255\n";

	for (unsigned int i = 0; i < nx * ny; ++i)
	{
		// BGR to RGB
		// 2 = R;
		// 1 = G;
		// 0 = B;
		fileHandler
			<< static_cast<int>(255.99f * image[i].e[2]) << " "
			<< static_cast<int>(255.99f * image[i].e[1]) << " "
			<< static_cast<int>(255.99f * image[i].e[0]) << "\n";

		//std::cout << i;
	}

	std::cout << "File Saved" << std::endl;
	fileHandler.close();

	delete[] image;
	return 0;
}