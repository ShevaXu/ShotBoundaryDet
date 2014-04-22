
#include "shotboundarydetector.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/ml/ml.hpp>

#include "time.h"

ShotBoundaryDetector::ShotBoundaryDetector( const string &fileName )
{
	/*m_fileName = fileName;
	m_label = fileName.substr(0, fileName.find_last_of('.'));*/
	if(loadParams(fileName))
	{
		m_capture.open(m_fileURL);
		if (m_capture.isOpened())
		{
			m_nFrames = (int)m_capture.get(CV_CAP_PROP_FRAME_COUNT);
		}
		else
			cout << "Can not open video file!\n";
		m_decodeTime = 0;
	}
	//
	
	//// default params
	//p_range = 50;
	//p_dcon = 10;
	////p_theta = 150.0;
	//p_tCut = 2.8;
	////p_tGT = 2.9;
	//p_tGTBegin = 3.1;
	//p_tGTEnd = 2.9;
	////p_tGphBegin = 3.1;
	////p_tGphEnd = 2.9;
	//p_GTMin = 5;
	//p_fusionType = SCS_FUSION_EARLY;
	// timer
		
}

int ShotBoundaryDetector::predict()
{
	// The train_data must have the CV_32FC1 (32-bit floating-point, single-channel), so does the row sample
	clock_t bg = clock();
	process();
	//
	m_predictVec.create(1, m_nFrames, CV_64FC1);
	// FOI predict
	if (p_withFOI)
	{
		if (!FOIPredict())
			return 0;
	}	
	// CUT predict
	if (!cutPredict())
		return 0;
	//// GT predict
	if (!GTPredict(p_fusionType))
		return 0;	
	/*if (!GPHPredict(fusionType))
		return 0;*/

	m_runTime = (int)(clock() - bg);
	outputXMLResult();
	return 1;
}

//int ShotBoundaryDetector::predict( const string &modelName, const string &scoreFile )
//{
//	return 0;
//}

//int ShotBoundaryDetector::getFrameVectors()
//{
//
//}

int ShotBoundaryDetector::process()
{
	if (m_nFrames > 100000)
	{
		cout << "The frame count may be wrong!\n";
	}
	else
		cout << "Frames to process: " << m_nFrames << endl;

	// step 1: get frame vectors
	cout << "Get frame histogram vectors ...\n";
	int frameCount = 0;
	Mat frame;
	std::vector<double> lumAvgs;
	while (true)
	{
		clock_t bg = clock();
		m_capture >> frame;	// input
		m_decodeTime += (int)(clock() - bg);
		if (frame.empty())
		{
			break;	// the end
		}
		frameCount++;
		// processing begin
		Mat temp = getFrameVector(frame);
		m_frameVecList.push_back(temp);
		Mat gray;
		cvtColor(frame, gray, CV_RGB2GRAY);
		Scalar avg = mean(gray);
		lumAvgs.push_back(avg[0]);
	}
	m_lumAvgVec = Mat(lumAvgs, true);	// make matrix
	//
	if (m_nFrames != frameCount)
	{
		m_nFrames = frameCount;
		cout << "Frame property is wrong! Reset to " << frameCount << endl;
	}

	// step 2: calculate weights
	cout << "Calculate weights ...\n";
	calculateWeights();

	// step 2: calculate scores
	cout << "Calculate scores ...\n";
	calculateScores();
	if (m_outputScore)
	{
		OutputScore();
	}

	return 1;
}

Mat ShotBoundaryDetector::getFrameVector( const Mat &frame )
{
	Mat vec(4 * 48, 1, CV_32FC1);
	int w = frame.cols / 2;
	int h = frame.rows / 2;
	int x[4] = {0, w, 0, w};
	int y[4] = {0, 0, h, h};
	for (int i = 0; i < 4; i++)
	{				
		Mat roi = frame(Rect(x[i], y[i], w, h));
		// get histogram for ROI
		Mat hist_c1, hist_c2, hist_c3;
		// Separate the image in 3 places ( B, G and R )
		std::vector<Mat> bgr_planes;
		split( roi, bgr_planes );
		int histSize = 16;
		// Set the ranges ( for B,G,R) )
		float range[] = { 0, 256 } ;
		const float* histRange = { range };
		// Compute the histograms:
		calcHist( &bgr_planes[0], 1, 0, Mat(), hist_c1, 1, &histSize, &histRange);
		calcHist( &bgr_planes[1], 1, 0, Mat(), hist_c2, 1, &histSize, &histRange);
		calcHist( &bgr_planes[2], 1, 0, Mat(), hist_c3, 1, &histSize, &histRange);
		// Normalize the result separately
		normalize(hist_c1, hist_c1, 1.0, 0, NORM_L1, -1, Mat() );
		normalize(hist_c2, hist_c2, 1.0, 0, NORM_L1, -1, Mat() );
		normalize(hist_c3, hist_c3, 1.0, 0, NORM_L1, -1, Mat() );
		// concencate hist		
		int startIdx = i * 48;
		hist_c1.copyTo(vec(Range(0 + startIdx, histSize + startIdx), Range::all()));
		hist_c2.copyTo(vec(Range(histSize + startIdx, histSize * 2 + startIdx), Range::all()));
		hist_c3.copyTo(vec(Range(histSize * 2 + startIdx, histSize * 3 + startIdx), Range::all()));
	}
	//
	return vec;
}

int ShotBoundaryDetector::calculateWeights()
{	
	m_weightM.create(m_nFrames, p_range, CV_64FC1);
	double temp, decayF;
	for (int i = 0; i < m_nFrames - p_range; i++)
	{
		for (int j = i + 1, k = 0; j <= i + p_range; j++, k++)
		{
			// calculate w(i,j)
			//temp = calIntersection(vecList[i], vecList[j]);
			temp = compareHist(m_frameVecList[i], m_frameVecList[j], CV_COMP_INTERSECT);	// histogram intersection
			/*if (temp != -1.0)
				cvmSet(wVMat, i, k, temp * getDecayF(i, j));
			else 
			{
				printf("Intersection error!\n");
				return -1.0;
			}*/
			decayF = j - i > p_range ? 0.0 : pow(1 - 0.02 * (j - i), 2.0);
			//return exp((float)((j-i)*(i-j))/(ZITA*ZITA));
			//return exp((float)(i-j)/ZITA);
			m_weightM.at<double>(i, k) = temp * decayF;
		}
	}
	for (int i = m_nFrames - p_range; i < m_nFrames - 1; i++)
	{
		for (int j = i + 1, k = 0; j < m_nFrames; j++, k++)
		{
			// calculate w(i,j)
			temp = compareHist(m_frameVecList[i], m_frameVecList[j], CV_COMP_INTERSECT);	// histogram intersection
			decayF = j - i > p_range ? 0.0 : pow(1 - 0.02 * (j - i), 2.0);
			m_weightM.at<double>(i, k) = temp * decayF;
		}
	}
	//
	return 1;
}

int ShotBoundaryDetector::calculateScores()
{
	m_scoresM.create(3, m_nFrames, CV_64FC1);
	for (int ss = 0; ss < 3; ss++)
	{
		//Mat scoreM = m_scoresM.row(ss);
		int gap = ss * 2 + 1;
		double temp;
		for (int t = p_range; t < m_nFrames - p_range; t++)
		{
			double assocA = 0.0, assocB = 0.0, cutAB = 0.0;
			// calculate assoc(A), A includes t
			for (int i = t - (p_dcon - 1) * gap; i < t; i += gap)
			{
				for (int j = i + gap; j <= t; j += gap)
				{
					//assocA += cvmGet(wVMat, i, j - i - 1);
					assocA += m_weightM.at<double>(i, j - i - 1);
				}
			}
			// calculate assoc(B), B excludes t
			for (int i = t + gap; i < t + p_dcon * gap; i += gap)
			{
				for (int j = i + gap; j <= t + p_dcon * gap; j += gap)
				{
					//assocB += cvmGet(wVMat, i, j - i - 1);
					assocB += m_weightM.at<double>(i, j - i - 1);
				}
			}
			// calculate cut(A,B)
			for (int i = t - (p_dcon - 1) * gap; i <= t; i += gap)
			{
				for (int j = t + gap; j <= t + p_dcon * gap; j += gap)
				{
					if (j - i - 1 < p_range)
						//cutAB += cvmGet(wVMat, i, j - i - 1);					
						cutAB += m_weightM.at<double>(i, j - i - 1);
				}
			}
			// calculate score_cut(A,B)
			if (assocA == 0.0 || assocB == 0.0)
			{
				//return -1.0;
				temp = 0.0;
			}
			else
				temp = cutAB / assocA + cutAB / assocB;			
			//
			m_scoresM.at<double>(ss, t) = temp;
		}
	}
	//
	return 1;
}

int ShotBoundaryDetector::OutputScore()
{
	string fileName = "output\\" + m_label + "_score.xml";
	FileStorage fs(fileName, FileStorage::WRITE);
	fs << "nFrames" << m_nFrames;
	fs << "Scores" << m_scoresM;
	fs << "LumAvgVec" << m_lumAvgVec;
	fs.release();
	return 1;
}

int ShotBoundaryDetector::loadScore( string fileName )
{
	FileStorage fs(fileName, FileStorage::READ);
	int temp = (int)fs["nFrames"];
	if (temp != m_nFrames)
	{
		// TODO
	}
	fs["Scores"] >> m_scoresM;
	fs["LumAvgVec"] >> m_lumAvgVec;
	fs.release();
	return 1;
}

int ShotBoundaryDetector::outputXMLResult()
{
	if (m_predictVec.empty())
	{
		cout << "The prediction is invalid!\n";
		return 0;
	}

	string url = "output\\" + m_label + "_result.xml";
	FILE *fout = fopen(url.c_str(),"w");

	fprintf(fout, "<!DOCTYPE shotBoundaryResult SYSTEM \"shotBoundaryResult.dtd\">\n");
	fprintf(fout, "<shotBoundaryResult sysId=\"sbd\" totalRunTime=\"%d\" totalDecodeTime=\"%d\" totalSegmentationTime=\"%d\" processorTypeSpeed=\"i5 2.7GHz\">\n",
		m_runTime, m_decodeTime, m_runTime - m_decodeTime);
	cout << "Runtime: " << m_runTime << " decodeTime: " << m_decodeTime << endl;
	//fprintf(fout, "<seg src=\"%s\" totalFNum=\"%d\">\n", m_fileName.c_str(), m_nFrames);
	fprintf(fout, "<seg src=\"%s\">\n", m_fileURL.c_str());
	double label, prelab;
	bool isGTBegin = 0;
	for (int i = 0; i < m_nFrames; i++)
	{
		//label = cvmGet(predictVec, 0, i);
		label = m_predictVec.at<double>(0, i);
		if (!isGTBegin)
		{
			if (label == 0.0)
				continue;
			if (label == 1.0) // cut
			{
				fprintf(fout, "<trans type=\"CUT\" preFNum=\"%d\" postFNum=\"%d\"/>\n", i, i+1);
				continue;
			}
			if (label == 2.0) // DIS
			{
				fprintf(fout, "<trans type=\"DIS\" preFNum=\"%d\"", i);
				isGTBegin = 1;
				prelab = 2.0;
			}
			if (label == 3.0) // FOI
			{
				fprintf(fout, "<trans type=\"FOI\" preFNum=\"%d\"", i);
				isGTBegin = 1;
				prelab = 3.0;
			}
			if (label == 4.0) // GPH
			{
				fprintf(fout, "<trans type=\"GPH\" preFNum=\"%d\"", i);
				isGTBegin = 1;
				prelab = 4.0;
			}
		}
		else
		{
			if (label != prelab)
			{
				fprintf(fout, " postFNum=\"%d\"/>\n", i-1);
				isGTBegin = 0;
			}
		}			
	}
	fprintf(fout, "</seg>\n</shotBoundaryResult>");
	fclose(fout);
	//
	return 1;
}

int ShotBoundaryDetector::activeSelect( int mode )
{
	if (mode == SCS_TYPE_CUT)
	{
		double temp, current;
		bool notMin;
		if (cutSelector.size() != 0)
			cutSelector.clear();
		// active select sample to avoid imbalance
		for (int t = p_range + p_dcon - 1; t < m_nFrames - p_range - p_dcon; t++)
		{
			//current =  cvmGet(score[0], 0, t);
			current = m_scoresM.at<double>(0, t);
			if (current > p_tCut) continue;
			// find local minimal
			notMin = 0;
			for (int k = t - p_dcon + 1; k <= t + p_dcon; k++)
			{
				//temp = cvmGet(score[0], 0, k);
				temp = m_scoresM.at<double>(0, k);
				if (temp < current)
				{
					notMin = 1;
					break;
				}
			}			
			if (!notMin)
			{	
				cutSelector.push_back(t);
			}			
		}
		cout << "Total " << cutSelector.size() << " cuts selected.\n";
		return 1;
	}
	if (mode == SCS_TYPE_DIS || mode == SCS_TYPE_GT)
	{
		double temp, current;
		//int s_begin = 0, s_end = 0, s_center;
		trans workTran;
		bool findEnd = 0;
		// active select sample to avoid imbalance
		for (int t = p_range + p_dcon - 1; t < m_nFrames - p_range - p_dcon; t++)
		{
			//temp = cvmGet(score[0], 0, t);
			temp = m_scoresM.at<double>(0, t);
			if (!findEnd) // find begin
			{					
				if (temp < p_tGTBegin)
				{
					workTran.preFNum = t;
					//workTran.preV = temp;
					findEnd = 1;
					//s_begin = t;
					//current = temp;
					//s_center = t;
				}
			}
			else // find end
			{
				if (temp < p_tGTEnd) // not the end
				{
					//if (temp < current) // change center
					//{
					//	s_center = t;
					//	current = temp;
					//}
					continue;
				}
				else // found the end
				{
					findEnd = 0;
					workTran.postFNum = t - 1;
					//workTran.postV = temp;						
					if (workTran.postFNum - workTran.preFNum >= p_GTMin)
					{
						workTran.selFNum = (workTran.postFNum + workTran.preFNum) / 2;
						GTSelector.push_back(workTran);
					}
				}
			}
		}
		cout << "Total " << GTSelector.size() << " GTs selected.\n";
		////selector refine
		//int nsgt = GTSelector.size();
		//for (int j = 0; j < nsgt - 1; j++)
		//{
		//	if (GTSelector[j+1].preFNum - GTSelector[j].postFNum < 30)
		//	{
		//		// merge two trans
		//		GTSelector[j].postFNum = GTSelector[j+1].postFNum;
		//		GTSelector[j].selFNum = (GTSelector[j].preFNum + GTSelector[j].postFNum) / 2;
		//		GTSelector[j+1] = GTSelector[j];
		//	}
		//}
		return 1;
	}
	return 0;
}

int ShotBoundaryDetector::cutPredict()
{
	if (m_scoresM.empty())
	{
		cout << "NO score matrix available!\n";
		return 0;
	}
	//
	if (!activeSelect(SCS_TYPE_CUT))
		return 0;
	//
	Mat rowSample(1, 2 * p_dcon, CV_32FC1);
	int nsel = cutSelector.size(), fNum, label, ncut = 0;
	double temp;
	SVM svm;
	svm.load(m_cutModel.c_str());
	//
	for (int ss = 0; ss < nsel; ss++)
	{
		fNum = cutSelector[ss];
		for (int j = 0, k = fNum - p_dcon + 1; k <= fNum + p_dcon; k++, j++)
		{			
			rowSample.at<float>(0, j) = (float)m_scoresM.at<double>(0, k);		
		}
		label = (int)svm.predict(rowSample, 0);
		if (label == 1)
		{
			//if (cvmGet(predictVec, 0, fNum) == 3.0) // overlapped with foi
			//{
			//	continue;
			//}
			ncut++;
			//temp = cvmGet(score[0], 0, fNum);
			//fprintf(fout, "%d:%f\n", fNum, temp);
			m_predictVec.at<double>(0, fNum) = 1.0;
		}
	}
	//
	cout << "Total " << ncut << " cuts detected!\n";
	//
	return 1;
}

int ShotBoundaryDetector::FOIPredict()
{
	// lumAvgVec: nFrames * 1
	if (m_lumAvgVec.empty())
	{
		cout << "No luminance vector available!\n";
		return 0;
	}
	int n = m_lumAvgVec.rows;
	for (int i = p_range; i < n - p_range; i++)
	{
		if (m_lumAvgVec.at<double>(i, 0) < p_tFOI)
		{
			/*for (int j = i - 20; j < i + 20; j++)
			{
				cvmSet(predictVec, 0, j, 3.0);
			}*/
			m_predictVec(Range::all(), Range(i - 20, i + 20)) = Scalar(3.0);
		}
	}
	//
	m_predictVec.at<double>(0, m_nFrames - 1) = 0.0;
	//
	return 1;
}

int ShotBoundaryDetector::GTPredict( int fusionType )
{
	if (m_scoresM.empty())
	{
		cout << "NO score matrix available!\n";
		return 0;
	}
	//
	if (!activeSelect(SCS_TYPE_GT))
		return 0;
	//			
	double temp;
	trans workTran;
	int nsel = GTSelector.size(), fNum, label, nGT = 0, olap = 0, enterp = 0;
	//int s_center, s_begin, s_end;
	if (fusionType == SCS_FUSION_EARLY)
	{
		//svm->load("model\\GTModel.xml", 0);
		SVM svm;
		svm.load(m_GTCModel.c_str());
		Mat rowSample(1, 2 * p_dcon * 3, CV_32FC1);
		for (int ss = 0; ss < nsel; ss++)
		{
			workTran = GTSelector[ss];
			fNum = workTran.selFNum;
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0, k = fNum - p_dcon + 1; k <= fNum + p_dcon; k++, j++)
				{
					/*temp = cvmGet(score[i], 0, k);
					cvmSet(rowsample, 0, j + i * 2 * DCON, temp);*/
					rowSample.at<float>(0, j + i * 2 * p_dcon) = (float)m_scoresM.at<double>(i, k);;
				}
			}
			label = (int)svm.predict(rowSample, 0);
			if (label == 1)
			{
				enterp++;
				bool overlap = 0;
				for (int m = workTran.preFNum; m <= workTran.postFNum; m++)
				{						
					//if (cvmGet(predictVec, 0, m) == 1.0 || cvmGet(predictVec, 0, m) == 3.0) // cut or foi overlapped
					double pred = m_predictVec.at<double>(0, m);
					if (pred == 1.0 || pred == 3.0) // cut or foi overlapped
					{
						overlap = 1;							
						break;
					}
				}
				if (overlap)
				{
					olap++;
					continue;
				}
				//
				nGT++;							
				//fprintf(fout, "%d:(%d-%d)\n", fNum, workTran.preFNum, workTran.postFNum);
				for (int m = workTran.preFNum; m <= workTran.postFNum; m++)
				{
					//cvmSet(predictVec, 0, m, 2.0);
					m_predictVec.at<double>(0, m) = 2.0;
				}
			}
		}			
	}
	else // fusionType == SCS_FUSION_LATE
	{
		Mat rowSample(1, 2 * p_dcon, CV_32FC1);
		SVM *lfsvm[3];
		lfsvm[0] = new SVM();
		lfsvm[0]->load(m_GTM_S1.c_str());
		lfsvm[1] = new SVM();
		lfsvm[1]->load(m_GTM_S2.c_str());
		lfsvm[2] = new SVM();
		lfsvm[2]->load(m_GTM_S3.c_str());
		for (int ss = 0; ss < nsel; ss++)
		{
			workTran = GTSelector[ss];
			fNum = workTran.selFNum;
			int decision = 0;
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0, k = fNum - p_dcon + 1; k <= fNum + p_dcon; k++, j++)
				{
					/*temp = cvmGet(score[i], 0, k);
					cvmSet(rowsample, 0, j, temp);*/
					rowSample.at<float>(0, j) = (float)m_scoresM.at<double>(i, k);
				}
				//Mat rowSample = m_scoresM(Rect(fNum - p_dcon + 1, i, p_dcon * 2, 1));
				label = (int)lfsvm[i]->predict(rowSample, 0);
				// if any one of the 3 SVM predicts it positive, it is positive 
				// now change to 2 !!!					
				if (label == 1)
				{
					decision++;
					//enterp++;
					//bool overlap = 0;
					//for (int m = workTran.preFNum; m <= workTran.postFNum; m++)
					//{						
					//	if (cvmGet(predictVec, 0, m) == 1.0 || cvmGet(predictVec, 0, m) == 3.0) // cut or foi overlapped
					//	{
					//		overlap = 1;							
					//		break;
					//	}
					//}
					//if (overlap)
					//{
					//	olap++;
					//	break;
					//}
					////
					//nGT++;							
					////fprintf(fout, "%d:(%d-%d)\n", fNum, workTran.preFNum, workTran.postFNum);
					//for (int m = workTran.preFNum; m <= workTran.postFNum; m++)
					//{
					//	cvmSet(predictVec, 0, m, 2.0);
					//}
					//break;
				}
			}
			if (decision >= p_tLateFusion)
			{
				enterp++;
				bool overlap = 0;
				for (int m = workTran.preFNum; m <= workTran.postFNum; m++)
				{						
					double pred = m_predictVec.at<double>(0, m);
					//if (cvmGet(predictVec, 0, m) == 1.0 || cvmGet(predictVec, 0, m) == 3.0) // cut or foi overlapped
					if (pred == 1.0 || pred == 3.0) // cut or foi overlapped
					{
						overlap = 1;							
						break;
					}
				}
				if (overlap)
				{
					olap++;
					continue;
				}
				//
				nGT++;							
				//fprintf(fout, "%d:(%d-%d)\n", fNum, workTran.preFNum, workTran.postFNum);
				for (int m = workTran.preFNum; m <= workTran.postFNum; m++)
				{
					//cvmSet(predictVec, 0, m, 2.0);
					m_predictVec.at<double>(0, m) = 2.0;
				}					
			}
		}			
	}
	//
	cout << "Total " << enterp << " predicted as gt!\n";
	cout << "Total " << olap << " overlapped!\n";
	//	
	cout << "Total " << nGT << " GT detected!\n";
	return 1;
}