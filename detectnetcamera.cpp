/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gstCamera.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "detectNet.h"


#define DEFAULT_CAMERA 0	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)
		

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
    printf("detectnet-camera\n  args (%i):  ", argc);


    std::string str = "pednet";
    argc = 1;
    std::strcpy(argv[0],str.c_str());

    for( int i=0; i < argc; i++ )
            printf("%i [%s]  ", i, argv[i]);

    printf("\n\n");

    if( signal(SIGINT, sig_handler) == SIG_ERR )
            printf("\ncan't catch SIGINT\n");


    /*
     * create the camera device
     */
    gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);

    if( !camera )
    {
            printf("\ndetectnet-camera:  failed to initialize video device\n");
            return 0;
    }

    printf("\ndetectnet-camera:  successfully initialized video device\n");
    printf("    width:  %u\n", camera->GetWidth());
    printf("   height:  %u\n", camera->GetHeight());
    printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());


    /*
     * create detectNet
     */
    detectNet* net = detectNet::Create(argc, argv);

    if( !net )
    {
            printf("detectnet-camera:   failed to initialize imageNet\n");
            return 0;
    }


    /*
     * allocate memory for output bounding boxes and class confidence
     */
    const uint32_t maxBoxes = net->GetMaxBoundingBoxes();
    const uint32_t classes  = net->GetNumClasses();

    float* bbCPU    = NULL;
    float* bbCUDA   = NULL;
    float* confCPU  = NULL;
    float* confCUDA = NULL;

    if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
        !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
    {
            printf("detectnet-console:  failed to alloc output memory\n");
            return 0;
    }


    /*
     * start streaming
     */
    if( !camera->Open() )
    {
            printf("\ndetectnet-camera:  failed to open camera for streaming\n");
            return 0;
    }

    printf("\ndetectnet-camera:  camera open for streaming\n");


    /*
     * processing loop
     */
    float confidence = 0.0f;

    cv::Mat cv_img(camera->GetHeight(),camera->GetWidth(),CV_8UC3);

    // init ros
    ros::init(argc, argv, "cameradetect");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("camera/detect", 1);

    sensor_msgs::ImagePtr msg;

    ros::Rate loop_rate(5);

    while( nh.ok() )
    {


            void* imgCPU  = NULL;
            void* imgCUDA = NULL;

            // get the latest frame
            if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
                    printf("\ndetectnet-camera:  failed to capture frame\n");

            // copy image
            memcpy((char*)cv_img.data,(char*)imgCPU,camera->GetHeight()*camera->GetWidth()*sizeof(uchar3));


            // convert from YUV to RGBA
            void* imgRGBA = NULL;

            if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
                    printf("detectnet-camera:  failed to convert from NV12 to RGBA\n");



            // classify image with detectNet
            int numBoundingBoxes = maxBoxes;

            if( net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU))
            {
                    //printf("%i bounding boxes detected\n", numBoundingBoxes);

                    int lastClass = 0;
                    int lastStart = 0;

                    for( int n=0; n < numBoundingBoxes; n++ )
                    {
                            const int nc = confCPU[n*2+1];
                            float* bb = bbCPU + (n * 4);

                            //printf("detected obj %i  class #%u (%s)  confidence=%f\n", n, nc, net->GetClassDesc(nc), confCPU[n*2]);
                            //printf("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]);

                            if(!cv_img.empty()){
                                cv::Rect rect((int)bb[0],(int)bb[1],(int)(bb[2]-bb[0]),(int)(bb[3]-bb[1]));
                                cv::rectangle(cv_img,rect,cv::Scalar(255,0,0),5,8,0);

                            }

                            if( nc != lastClass || n == (numBoundingBoxes - 1) )
                            {

                                    lastClass = nc;
                                    lastStart = n;

                                    CUDA(cudaDeviceSynchronize());
                            }
                    }


            }


            if(!cv_img.empty()){
                cv::cvtColor(cv_img,cv_img,CV_RGB2BGR);
                msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_img).toImageMsg();
                pub.publish(msg);
            }


            ros::spinOnce();
            loop_rate.sleep();



    }

    printf("\ndetectnet-camera:  un-initializing video device\n");


    /*
     * shutdown the camera device
     */
    if( camera != NULL )
    {
            delete camera;
            camera = NULL;
    }


    printf("detectnet-camera:  video device has been un-initialized.\n");
    printf("detectnet-camera:  this concludes the test of the video device.\n");
    return 0;
}

