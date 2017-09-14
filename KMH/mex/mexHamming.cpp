#define DLL_EXPORT_SYM

#include <algorithm>
//#define __STDC_UTF_16__
#include "mex.h"
#include <intrin.h>
#include <immintrin.h>

#define IN_BincodeXtraining prhs[0]
#define IN_BincodeXtest prhs[1]
#define IN_gt prhs[2]
#define IN_K prhs[3]
#define IN_sel prhs[4]
#define OUT_recall_vs_sel plhs[0]

typedef unsigned char byte;

struct HammingDistIdx
{
	int nDist;
	int nIdx;
};

bool compare_dist(HammingDistIdx item_a, HammingDistIdx item_b) { return (item_a.nDist < item_b.nDist); }

void hamming_search_32bits(int* pNeighborIdx, const byte* pDataBin, const byte* pQueryBin,
	const int nNumData, const int nNumQueries, const int nSelectivity)
{
	typedef unsigned int EntryType;	// 32bits
	const int nBytes = 4;

	EntryType* pData;
	EntryType* pQuery;

	HammingDistIdx* pHammingDist = (HammingDistIdx*)_aligned_malloc(nNumData * sizeof(HammingDistIdx), 16);

	/// reset query pointer
	pQuery = (EntryType*)pQueryBin;

	for (int idx_query = 0; idx_query < nNumQueries; idx_query ++)
	{
		/// reset data pointer
		pData = (EntryType*)pDataBin;

		/// start scan data points
		for (int idx_data = 0; idx_data < nNumData; idx_data ++)
		{			
			pHammingDist[idx_data].nDist = (int)__popcnt((*pQuery) ^ (*pData++));
			pHammingDist[idx_data].nIdx = idx_data;
		}//idx_data

		std::partial_sort(pHammingDist, pHammingDist + nSelectivity, pHammingDist + nNumData, compare_dist);

		for (int i = 0; i < nSelectivity; i ++)
			*pNeighborIdx ++ = pHammingDist[i].nIdx;

		pQuery ++;
	}
	
	_aligned_free(pHammingDist);
}

void hamming_search_64bits(int* pNeighborIdx, const byte* pDataBin, const byte* pQueryBin,
	const int nNumData, const int nNumQueries, const int nSelectivity)
{
	typedef unsigned __int64 EntryType;	// 64bits
	const int nBytes = 8;

	EntryType* pData;
	EntryType* pQuery;

	HammingDistIdx* pHammingDist = (HammingDistIdx*)_aligned_malloc(nNumData * sizeof(HammingDistIdx), 16);

	/// reset query pointer
	pQuery = (EntryType*)pQueryBin;

	for (int idx_query = 0; idx_query < nNumQueries; idx_query ++)
	{
		/// reset data pointer
		pData = (EntryType*)pDataBin;

		/// start scan data points
		for (int idx_data = 0; idx_data < nNumData; idx_data ++)
		{	
			pHammingDist[idx_data].nDist = (int)__popcnt64((*pQuery) ^ (*pData++));
			pHammingDist[idx_data].nIdx = idx_data;
		}//idx_data

		std::partial_sort(pHammingDist, pHammingDist + nSelectivity, pHammingDist + nNumData, compare_dist);

		for (int i = 0; i < nSelectivity; i ++)
			*pNeighborIdx ++ = pHammingDist[i].nIdx;

		pQuery ++;
	}
	
	_aligned_free(pHammingDist);
}

void hamming_search_128bits(int* pNeighborIdx, const byte* pDataBin, const byte* pQueryBin,
	const int nNumData, const int nNumQueries, const int nSelectivity)
{
	typedef __m128i EntryType;	// 128bits
	const int nBytes = 16;

	EntryType* pData;
	EntryType* pQuery;

	HammingDistIdx* pHammingDist = (HammingDistIdx*)_aligned_malloc(nNumData * sizeof(HammingDistIdx), 16);

	/// reset query pointer
	pQuery = (EntryType*)pQueryBin;

	for (int idx_query = 0; idx_query < nNumQueries; idx_query ++)
	{
		/// reset data pointer
		pData = (EntryType*)pDataBin;

		unsigned __int64 *pQuerySec0 = (unsigned __int64*)pQuery;
		unsigned __int64 *pQuerySec1 = pQuerySec0 + 1;

		/// start scan data points
		for (int idx_data = 0; idx_data < nNumData; idx_data ++)
		{			
			unsigned __int64 *pDataSec0 = (unsigned __int64*)pData;
			unsigned __int64 *pDataSec1 = pDataSec0 + 1;

			pHammingDist[idx_data].nDist = (int)__popcnt64(*pDataSec0 ^ *pQuerySec0) + (int)__popcnt64(*pDataSec1 ^ *pQuerySec1);
			pHammingDist[idx_data].nIdx = idx_data;

			pData ++;
		}//idx_data

		std::partial_sort(pHammingDist, pHammingDist + nSelectivity, pHammingDist + nNumData, compare_dist);

		for (int i = 0; i < nSelectivity; i ++)
			*pNeighborIdx ++ = pHammingDist[i].nIdx;

		pQuery ++;
	}
	
	_aligned_free(pHammingDist);
}

void compute_precision(byte* pNumSuccess, const int* pNeighborIdxTruth, const int* pNeighborIdxHashing,
	const int nNumQueries, const int nTruthListSize, const int nHashingListSize, const int nNumRetrievedPoints, const int nKNN)
{
	const int* pNeighborListTruth = pNeighborIdxTruth;
	const int* pNeighborListHashing = pNeighborIdxHashing;

	memset((void*)pNumSuccess, 0, nNumQueries * nNumRetrievedPoints * sizeof(byte));

	for (int idx_query = 0; idx_query < nNumQueries; idx_query ++)
	{	
		for (int i = 0; i < nNumRetrievedPoints; i ++)
		{
			const int idx_hashing = pNeighborListHashing[i];
			bool bSuccess = false;
			for (int j = 0; j < nKNN; j ++)
			{
				if (idx_hashing == pNeighborListTruth[j])	// success
				{
					bSuccess = true;
					
					break;
				}
			}//j

			if (bSuccess) pNumSuccess[i] = 1;
		}//i
		
		pNeighborListTruth += nTruthListSize;
		pNeighborListHashing += nHashingListSize;

		pNumSuccess += nNumRetrievedPoints;

	}//idx_query
	printf("Evaluate %d queries: done\n", nNumQueries);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	const int nNumData = (int)mxGetN(IN_BincodeXtraining);
	const int nNumQueries = (int)mxGetN(IN_BincodeXtest);
	const int nBytes = (int)mxGetM(IN_BincodeXtraining);
	const int nSelectivity = (int)mxGetScalar(IN_sel);
	const int nBits = nBytes * 8;

	byte* pDataBin = (byte*)mxGetPr(IN_BincodeXtraining);
	byte* pQueryBin = (byte*)mxGetPr(IN_BincodeXtest);

	printf("nNumData: %d\n", nNumData);
	printf("nNumQueries: %d\n", nNumQueries);
	printf("nBytes: %d\n", nBytes);
	printf("nSelectivity: %d\n", nSelectivity);

	int* pNeighborIdxHashing = (int*)malloc(nNumQueries * nSelectivity * sizeof(int));

	//// hamming ranking

	if (nBits == 32)
	{
		//hamming_search_32bits(pNeighborIdxHashing, pDataBin, pQueryBin, nNumData, nNumQueries, nSelectivity);
		
		const int nNumCores = 8;
		const int nSubNumQueries = nNumQueries / nNumCores;

		#pragma omp parallel for
		for (int core = 0; core < nNumCores; core ++)
		{
			hamming_search_32bits(pNeighborIdxHashing + core * nSubNumQueries * nSelectivity,
				pDataBin, pQueryBin + core * nSubNumQueries * nBytes, nNumData, nSubNumQueries, nSelectivity);
		}//core
	}
	else if (nBits == 64)
	{
		//hamming_search_64bits(pNeighborIdxHashing, pDataBin, pQueryBin, nNumData, nNumQueries, nSelectivity);
		
		const int nNumCores = 8;
		const int nSubNumQueries = nNumQueries / nNumCores;

		#pragma omp parallel for
		for (int core = 0; core < nNumCores; core ++)
		{
			hamming_search_64bits(pNeighborIdxHashing + core * nSubNumQueries * nSelectivity,
				pDataBin, pQueryBin + core * nSubNumQueries * nBytes, nNumData, nSubNumQueries, nSelectivity);
		}//core
	}
	else if (nBits == 128)
	{
		//hamming_search_64bits(pNeighborIdxHashing, pDataBin, pQueryBin, nNumData, nNumQueries, nSelectivity);
		
		const int nNumCores = 8;
		const int nSubNumQueries = nNumQueries / nNumCores;

		#pragma omp parallel for
		for (int core = 0; core < nNumCores; core ++)
		{
			hamming_search_128bits(pNeighborIdxHashing + core * nSubNumQueries * nSelectivity,
				pDataBin, pQueryBin + core * nSubNumQueries * nBytes, nNumData, nSubNumQueries, nSelectivity);
		}//core
	}
	else
	{
		printf("False bit number.\n");
		return;
	}

	//// evaluation
	const int nTruthListSize = (int)mxGetM(IN_gt);
	const int nKNN = (int)mxGetScalar(IN_K);

	int* pNeighborIdxTruth = (int*)mxGetPr(IN_gt);

	printf("nKNN: %d\n", nKNN);

	byte* pNumSuccess = (byte*)malloc(nNumQueries * nSelectivity * sizeof(byte));
	compute_precision(pNumSuccess, pNeighborIdxTruth, pNeighborIdxHashing, nNumQueries, nTruthListSize, nSelectivity, nSelectivity, nKNN);
	
	OUT_recall_vs_sel = mxCreateDoubleMatrix(nSelectivity, 1, mxREAL);
	
	double* recall_vs_sel = mxGetPr(OUT_recall_vs_sel);

	for (int j = 0; j < nSelectivity; j ++)
		recall_vs_sel[j] = 0;
		
	for (int i = 0; i < nNumQueries; i ++)
	{
		byte* pNumSuccessCurrent = pNumSuccess + i * nSelectivity;

		for (int j = 0; j < nSelectivity; j ++)	
			recall_vs_sel[j] += (double)pNumSuccessCurrent[j];
	}
	
	for (int j = 0; j < nSelectivity; j ++)
		recall_vs_sel[j] /= nKNN * nNumQueries;

	for (int j = 1; j < nSelectivity; j ++)
		recall_vs_sel[j] = recall_vs_sel[j - 1] + recall_vs_sel[j];


	free(pNeighborIdxHashing);
	free(pNumSuccess);

	return;
}