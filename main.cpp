//#include "stdlib.h"

#include "shotboundarydetector.h"

int main()
{
	ShotBoundaryDetector detector("param.xml");
	detector.predict();

	return 1;
}