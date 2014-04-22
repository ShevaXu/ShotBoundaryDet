#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <iostream>

using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////
#define SCS_FUSION_EARLY 0
#define SCS_FUSION_LATE 1
#define SCS_TYPE_CUT 2
#define SCS_TYPE_DIS 3
#define SCS_TYPE_FOI 4
#define SCS_TYPE_GT 9


//////////////////////////////////////////////////////////////////////////
struct trans 
{
	int preFNum;
	int postFNum;
	int selFNum;
};

//////////////////////////////////////////////////////////////////////////
class ShotBoundaryDetector
{
private:

	VideoCapture m_capture;

	int m_nFrames;

	string m_fileURL,
		m_label;

	int m_decodeTime,
		m_runTime;

	std::vector<Mat> m_frameVecList;
	Mat m_weightM,
		m_scoresM,
		m_lumAvgVec,
		m_predictVec;

	// parameters
	int p_range,
		p_dcon;
	double p_tCut,
		p_tGTBegin,
		p_tGTEnd;
	int p_GTMin;
	int p_fusionType;
	int p_tLateFusion;
	bool p_withFOI,
		m_outputScore;
	int p_tFOI;

	// models
	string m_cutModel,
		m_GTCModel,
		m_GTM_S1,
		m_GTM_S2,
		m_GTM_S3;


	// selector
	std::vector<trans> GTSelector;
	std::vector<int> cutSelector;

public:

	ShotBoundaryDetector(const string &fileName);

	int process();

	int predict();

protected:

	// processing steps	
	Mat getFrameVector(const Mat &frame);
	int calculateWeights();
	int calculateScores();

	int activeSelect(int mode);

	int outputXMLResult();

	int OutputScore();
	int loadScore(string fileName);	

	// predict
	int cutPredict();
	int GTPredict(int fusionType);
	int FOIPredict();

	// 
	bool loadParams(string FileName)
	{
		FileStorage fs(FileName, FileStorage::READ);
		if (!fs.isOpened())
		{
			cout << "Fail to load param.xml!\n";
			return false;
		}
		//
		m_fileURL = fs["FileURL"];
		m_label = m_fileURL.substr(0, m_fileURL.find_last_of('.'));
		//
		p_range = (int)fs["p_weightRange"];
		p_dcon = (int)fs["p_scoreCuts"];
		p_tCut = (double)fs["p_threshCut"];
		p_tGTBegin = (double)fs["p_threshGTBegin"];
		p_tGTEnd = (double)fs["p_threshGTEnd"];
		p_GTMin = (int)fs["p_GTMinLength"];
		p_fusionType = (int)fs["p_fusionType"];
		p_withFOI = (bool)((int)fs["withFOI"]);
		p_tFOI = (int)fs["ThreshFOI"];
		p_tLateFusion = (int)fs["ThreshLateFusion"];
		//
		m_outputScore = (bool)((int)fs["outputScore"]);
		m_cutModel = fs["cutModelURL"];
		m_GTCModel = fs["GTCombinedModelURL"];
		m_GTM_S1 = fs["GTModel_S1"];
		m_GTM_S2 = fs["GTModel_S2"];
		m_GTM_S3 = fs["GTModel_S3"];
		//
		fs.release();
		return true;
	}
};