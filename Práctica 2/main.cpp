#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>
#include <stdlib.h>

using namespace cv;
const int NOOPER_MODE = -1;
const int DETECCION_JUG1 = 0;
const int DETECCION_JUG2 = 1;
const int DETECCION_CESPED = 2;
const int DETECCION_LINEAS = 3;
const int DETECCION_BALON = 4;
const int FUERA_JUEGO = 5;
const int TACTICA = 6;
const int COLORMODEL_MODE = 7;
const int MEDIAN_MODE = 8;
const int NORMHIST_MODE = 9;
const int SEGMENTATION_MODE = 10;

void redimensionado (Mat img) {
	Mat img_out;
	imwrite("resolucion_orig.png", img);
	resize(img, img_out, Size(0,0), 0.8, 0.8);
	imshow("image1", img_out);
	imwrite("resolucion_0,8.png", img_out);
	cvWaitKey(0);
	resize(img, img_out, Size(0,0), 0.5, 0.5);
	imshow("image1", img_out);
	imwrite("resolucion_0,5.png", img_out);
	cvWaitKey(0);
	resize(img, img_out, Size(0,0), 0.25, 0.25);
	imshow("image1", img_out);
	imwrite("resolucion_0,25.png", img_out);
	cvWaitKey(0);
	resize(img, img_out, Size(0,0), 0.1, 0.1);
	imshow("image1", img_out);
	imwrite("resolucion_0,1.png", img_out);
	cvWaitKey(0);
	resize(img, img_out, Size(0,0), 0.05, 0.05);
	imshow("image1", img_out);
	imwrite("resolucion_0,05.png", img_out);
	cvWaitKey(0);
}

void reduccionNivGris (Mat img) {
	Mat img_out, img_out2, img_out3;
	cvtColor(img, img_out, CV_BGR2GRAY);
	imwrite("niv_gris_original.png", img_out);
	double scale = (2.0 - 1.0)/256.0;
	img_out.convertTo(img_out2, 8, scale);
	img_out2.convertTo(img_out2, 8, 1.0/scale);
	imshow("image1", img_out2);
	imwrite("2_niv_gris.png", img_out2);
	cvWaitKey(0);
	scale = (4.0 - 1.0)/256.0;
	img_out.convertTo(img_out2, 8, scale);
	img_out2.convertTo(img_out2, 8, 1.0/scale);
	imshow("image1", img_out2);
	imwrite("4_niv_gris.png", img_out2);
	cvWaitKey(0);
	scale = (8.0 - 1.0)/256.0;
	img_out.convertTo(img_out2, 8, scale);
	img_out2.convertTo(img_out2, 8, 1.0/scale);
	imshow("image1", img_out2);
	imwrite("8_niv_gris.png", img_out2);
	cvWaitKey(0);
	scale = (16.0 - 1.0)/256.0;
	img_out.convertTo(img_out2, 8, scale);
	img_out2.convertTo(img_out2, 8, 1.0/scale);
	imshow("image1", img_out2);
	imwrite("16_niv_gris.png", img_out2);
	cvWaitKey(0);
	scale = (32.0 - 1.0)/256.0;
	img_out.convertTo(img_out2, 8, scale);
	img_out2.convertTo(img_out2, 8, 1.0/scale);
	imshow("image1", img_out2);
	imwrite("32_niv_gris.png", img_out2);
	cvWaitKey(0);
	scale = (64.0 - 1.0)/256.0;
	img_out.convertTo(img_out2, 8, scale);
	img_out2.convertTo(img_out2, 8, 1.0/scale);
	imshow("image1", img_out2);
	imwrite("64_niv_gris.png", img_out2);
	cvWaitKey(0);
}

void aplicarThreshold (Mat img, string nombre) {
	Mat img_out, img_out2;
	cvtColor(img, img_out, CV_BGR2GRAY);
	threshold(img_out, img_out2, 60, 255, THRESH_BINARY);
	imshow("image1", img_out2);
	imwrite(nombre + "_thr_60.png", img_out2);
	cvWaitKey(0);
	threshold(img_out, img_out2, 80, 255, THRESH_BINARY);
	imshow("image1", img_out2);
	imwrite(nombre + "_thr_80.png", img_out2);
	cvWaitKey(0);
	threshold(img_out, img_out2, 100, 255, THRESH_BINARY);
	imshow("image1", img_out2);
	imwrite(nombre + "_thr_100.png", img_out2);
	cvWaitKey(0);
	threshold(img_out, img_out2, 120, 255, THRESH_BINARY);
	imshow("image1", img_out2);
	imwrite(nombre + "_thr_120.png", img_out2);
	cvWaitKey(0);
	threshold(img_out, img_out2, 140, 255, THRESH_BINARY);
	imshow("image1", img_out2);
	imwrite(nombre + "_thr_140.png", img_out2);
	cvWaitKey(0);
}

void separarCanalesRGB (Mat img, string nombre) {
	Mat canal1(img.rows, img.cols, CV_8UC3);
	Mat canal2(img.rows, img.cols, CV_8UC3);
	Mat canal3(img.rows, img.cols, CV_8UC3);
	Mat img_out1[1] = {canal1};
	Mat img_out2[1] = {canal2};
	Mat img_out3[1] = {canal3};
	int from_to[] = {2,2};
	mixChannels( &img, 1, img_out1, 1, from_to, 1);
	imshow(nombre + "Red", img_out1[0]);
	imwrite(nombre + "Red.png", img_out1[0]);
	cvWaitKey(0);
	from_to[0] = 1; from_to[1] = 1;
	mixChannels( &img, 1, img_out2, 1, from_to, 1);
	imshow(nombre + "Green", img_out2[0]);
	imwrite(nombre + "Green.png", img_out2[0]);
	cvWaitKey(0);
	from_to[0] = 0; from_to[1] = 0;
	mixChannels( &img, 1, img_out3, 1, from_to, 1);
	imshow(nombre + "Blue", img_out3[0]);
	imwrite(nombre + "Blue.png", img_out3[0]);
	cvWaitKey(0);
}

void separarCanales (Mat img, string nombre) {
	Mat canal1(img.rows, img.cols, CV_8UC1);
	Mat canal2(img.rows, img.cols, CV_8UC1);
	Mat canal3(img.rows, img.cols, CV_8UC1);
	Mat img_out1[1] = {canal1};
	Mat img_out2[1] = {canal2};
	Mat img_out3[1] = {canal3};
	int from_to[] = {2,0};
	mixChannels( &img, 1, img_out1, 1, from_to, 1);
	imshow(nombre + "Canal3", img_out1[0]);
	imwrite(nombre + "Canal3.png", img_out1[0]);
	cvWaitKey(0);
	from_to[0] = 1; from_to[1] = 0;
	mixChannels( &img, 1, img_out2, 1, from_to, 1);
	imshow(nombre + "Canal2", img_out2[0]);
	imwrite(nombre + "Canal2.png", img_out2[0]);
	cvWaitKey(0);
	from_to[0] = 0; from_to[1] = 0;
	mixChannels( &img, 1, img_out3, 1, from_to, 1);
	imshow(nombre + "Canal1", img_out3[0]);
	imwrite(nombre + "Canal1.png", img_out3[0]);
	cvWaitKey(0);
}

void cambiarModelosColor (Mat img) {
	Mat img_out;
	separarCanalesRGB(img, "RGB_");
	cvtColor(img, img_out, CV_BGR2GRAY);
	imshow("GRAY", img_out);
	cvWaitKey(0);
	cvtColor(img, img_out, CV_BGR2HLS);
	imshow("HLS", img_out);
	imwrite("HLS.png", img_out);
	separarCanales(img_out, "HLS_");
	cvWaitKey(0);
	cvtColor(img, img_out, CV_BGR2XYZ);
	imshow("CIE XYZ", img_out);
	imwrite("CIEXYZ.png", img_out);
	separarCanales(img_out, "CIE XYZ_");
	cvWaitKey(0);
	cvtColor(img, img_out, CV_BGR2HSV);
	imshow("HSV", img_out);
	imwrite("HSV.png", img_out);
	separarCanales(img_out, "HSV_");
	cvWaitKey(0);
}

void desenfoqueGaussiano (Mat img, string nombre) {
	Mat img_out1, img_out2, img_out3;
	GaussianBlur(img, img_out1, Size(25,25), 1, 1);
	GaussianBlur(img, img_out2, Size(25,25), 3, 3);
	GaussianBlur(img, img_out3, Size(25,25), 5, 5);
	imshow("image1", img_out1);
	imwrite(nombre + "1.png", img_out1);
	cvWaitKey(0);
	imshow("image1", img_out2);
	imwrite(nombre + "2.png", img_out2);
	cvWaitKey(0);
	imshow("image1", img_out3);
	imwrite(nombre + "3.png", img_out3);
	cvWaitKey(0);
}

void filtroMediana (Mat img, string nombre) {
	Mat img_out1, img_out2, img_out3;
	medianBlur(img, img_out1, 1);
	medianBlur(img_out1, img_out2,  3);
	medianBlur(img_out2, img_out3, 5);
	imshow("image1", img_out1);
	imwrite(nombre + "1.png", img_out1);
	cvWaitKey(0);
	imshow("image1", img_out2);
	imwrite(nombre + "2.png", img_out2);
	cvWaitKey(0);
	imshow("image1", img_out3);
	imwrite(nombre + "3.png", img_out3);
	cvWaitKey(0);
}

void anadirRuido(Mat img)
{
	// Ruido gaussiano
	Mat img_channels[3];
	Mat img_out;
	split(img, img_channels);
	Mat gaussian_noise = img_channels[0].clone();
	Mat media = img_channels[0].clone();
	media.setTo(108);
	for (int i = 0; i < img.channels(); i++) {
		randn(gaussian_noise,128,30);
		img_channels[i] = (gaussian_noise - media) + img_channels[i];
		img_channels[i] = (media - gaussian_noise) + img_channels[i];
	}
	merge(img_channels, 3, img_out);
	imshow("Ruido gaussiano", img_out);
	imwrite("ruido_gaussiano.png", img_out);
	cvWaitKey(0);

	desenfoqueGaussiano(img_out, "ruido_gaussiano-filtro_gaussiano");
	filtroMediana(img_out, "ruido_gaussiano-filtro_mediana");
	
	// Ruido salt and pepper
	split(img, img_channels);
	Mat saltpepper_noise = Mat::zeros(img.rows, img.cols,CV_8U);
	randu(saltpepper_noise,0,255);
	Mat negro = saltpepper_noise < 12;
	Mat blanco = saltpepper_noise > 253;
	media.setTo(128);
	for (int i = 0; i < img.channels(); i++) {
		img_channels[i].setTo(255,blanco);
		img_channels[i].setTo(0,negro);
	}
	merge(img_channels, 3, img_out);
	imshow("Ruido salt and pepper", img_out);
	imwrite("ruido_saltandpepper.png", img_out);
	cvWaitKey(0);

	desenfoqueGaussiano(img_out, "ruido_saltandpepper-filtro_gaussiano");
	filtroMediana(img_out, "ruido_saltandpepper-filtro_mediana");
}

void manipulacionContraste(Mat img, string nombre)
{
	Mat img_out, img_grises;
	Mat img_channels[3];
	split(img, img_channels);
	cvtColor(img, img_grises, CV_BGR2GRAY);
	equalizeHist(img_grises, img_grises);
	imshow("image", img_grises);
	imwrite("manipul_contraste_gris_" + nombre + ".png",img_grises);
	cvWaitKey(0);
	for (int i = 0; i < img.channels(); i++) {
		equalizeHist(img_channels[i], img_channels[i]);
	}
	merge(img_channels, img.channels(), img_out);
	imshow("image", img_out);
	imwrite("manipul_contraste_color_" + nombre + ".png",img_out);
	cvWaitKey(0);
}

void realceContornos(Mat img)
{
	Mat img_blur, img_out1, img_out2, img_out3;
	GaussianBlur(img, img_blur, Size(9,9), 1.5, 1.5);
	img_out1 = img + 1 * (img - img_blur);
	img_out2 = img + 2 * (img - img_blur);
	img_out3 = img + 3 * (img - img_blur);
	imshow("image1", img_out1);
	cvWaitKey(0);
	imshow("image2", img_out2);
	cvWaitKey(0);
	imshow("image3", img_out3);
	cvWaitKey(0);
}

void aplicarSobel (Mat img, string nombre) {
	Mat img_out;
	int ddepth = CV_16S;
	cvtColor(img, img_out, CV_BGR2GRAY);;
	Mat grad;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Sobel( img_out, grad_x, ddepth, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	Sobel( img_out, grad_y, ddepth, 0, 1, 3, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
	imshow( "image1", grad );
	imwrite("sobel_" + nombre + ".png",grad);
	waitKey(0);
}

void aplicarLaplace (Mat img, string nombre) {
	Mat img_out;
	cvtColor(img, img_out, CV_BGR2GRAY);
	Laplacian(img_out, img_out, 8);
	imshow("image1", img_out);
	imwrite("laplace_" + nombre + ".png",img_out);
	cvWaitKey(0);
}

void aplicarCanny (Mat img, string nombre) {
	Mat img_out;
	cvtColor(img, img_out, CV_BGR2GRAY);
	Canny(img_out, img_out, 100, 200);
	imshow("image1", img_out);
	imwrite(nombre + ".png", img_out);
	cvWaitKey(0);
}

void transformadaHough(Mat img)
{
	Mat img_out1, img_out2;
	vector<Vec2f> lineas;
	float rho, theta;
	double a, b, x0, y0;
	cvtColor(img, img_out1, CV_BGR2GRAY);
	Canny(img_out1, img_out1, 100, 200);
	cvtColor(img_out1, img_out2, CV_GRAY2BGR);
	HoughLines(img_out1, lineas, 1, CV_PI/180, 60);
	for(size_t i = 0; i < lineas.size(); i++)
    {
        rho = lineas[i][0];
        theta = lineas[i][1];
        a = cos(theta);
		b = sin(theta);
        x0 = a * rho;
		y0 = b * rho;
        Point p1(cvRound(x0 + 1000 * (-b)),
                  cvRound(y0 + 1000 * (a)));
        Point p2(cvRound(x0 - 1000 * (-b)),
                  cvRound(y0 - 1000 * (a)));
        line(img_out2, p1, p2, Scalar(0,0,255), 1, 8);
    }
	imshow("image", img_out2);
	imwrite("image.png", img_out2);
	cvWaitKey(0);
	cvWaitKey(0);
}

void segmentacionUmbralizacion(Mat img)
{
	Mat img_out;
	cvtColor(img, img_out, CV_BGR2GRAY);
	adaptiveThreshold(img_out, img_out, 100, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 199, 5);
	imshow("image1", img_out);
	imwrite("imagen.png", img_out);
	cvWaitKey(0);
}

void segmentacionCrecimiento(Mat img){
	Mat img_out;
	cvtColor(img, img_out, CV_BGR2GRAY);
	Canny(img_out, img_out, 10, 250);
	imshow("image1", img_out);
	cvWaitKey(0);
	int n = 100;
	unsigned char colorActual = 50;
	Mat mascara = Mat::zeros(img.rows - 2, img.cols - 2,CV_8UC1);
	for (int i = 0; i < n; i++) {
		int x = rand() % mascara.cols;
		int y = rand() % mascara.rows;
		Point p(x,y);
		if (mascara.at<uchar>(p) == 0) {
			floodFill(mascara, img_out, p, colorActual, 0, Scalar(), Scalar(), 4);
			imshow("image1", mascara);
			cvWaitKey(0);
			colorActual += 50;
		}
	}
	
}

void segmentacionMorph (Mat img){
	Mat img_out;
	cvtColor(img, img_out, CV_BGR2GRAY);
	equalizeHist(img_out, img_out);
	adaptiveThreshold(img_out, img_out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 200,5);
	imshow("image1", img_out);
	imwrite("imagen.png", img_out);
	cvWaitKey(0);
	Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 7, 7 ),
                                       Point( 3, 3) );
	int  n = 1;
	for (int i = 0; i < n; i++) {
		erode(img_out, img_out, element);
		dilate(img_out, img_out, element);
		imshow("image1", img_out);
		cvWaitKey(0);
	}
}

// ----------------------


void pintarLineas (Mat img, vector<Vec2f> lineas) {
	float rho, theta;
	double a, b, x0, y0;
	for(size_t i = 0; i < lineas.size(); i++)
    {
        rho = lineas[i][0];
        theta = lineas[i][1];
        a = cos(theta);
		b = sin(theta);
        x0 = a * rho;
		y0 = b * rho;
        Point p1(cvRound(x0 + 1000 * (-b)),
                  cvRound(y0 + 1000 * (a)));
        Point p2(cvRound(x0 - 1000 * (-b)),
                  cvRound(y0 - 1000 * (a)));
        line(img, p1, p2, Scalar(0,0,255), 1, 8);
    }
}

void filtrarLineas (vector<Vec2f> lineas){
	
}

void pintarLineasP (Mat img, vector<Vec4i> lineas) {
	for( size_t i = 0; i < lineas.size(); i++ )
	{
		Vec4i l = lineas[i];
		line( img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
	}
}

void pintarCirculos (Mat img, vector<Vec3f> circles) {
	for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( img, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( img, center, radius, Scalar(0,0,255), 3, 8, 0 );
   }
}

void eliminarPaneles(Mat img) {
	rectangle(img, Point(62,36), Point(470, 71), CV_RGB(0,0,0), CV_FILLED);
	rectangle(img, Point(65,633), Point(65+241, 633+47), CV_RGB(0,0,0), CV_FILLED);
	rectangle(img, Point(982,633), Point(982+241, 633+47), CV_RGB(0,0,0), CV_FILLED);
	rectangle(img, Point(549,580), Point(728, 688), CV_RGB(0,0,0), CV_FILLED);
	imshow("image1", img);
	cvWaitKey(0);
}

// Equipo rojo
vector<vector<Point>> detectarJugadores1 (Mat img, string nombre) {
	Mat imgHSV, imgThresh;
	vector<vector<Point> > contours;

	cvtColor(img, imgHSV, CV_BGR2HSV);
	inRange(imgHSV, cvScalar(130,160,40), cvScalar(200,256,256), imgThresh);
	imshow("image1", img);
	cvWaitKey(0);
	imshow("image1", imgHSV);
	imwrite("detectar_jugadores_eq1_1_" + nombre + ".png", imgHSV);
	cvWaitKey(0);
	imshow("image1", imgThresh);
	imwrite("detectar_jugadores_eq1_2_" + nombre + ".png", imgThresh);
	cvWaitKey(0);
	eliminarPaneles(imgThresh);
	imshow("image1", imgThresh);
	imwrite("detectar_jugadores_eq1_3_" + nombre + ".png", imgThresh);
	Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 40, 40 ),
                                       Point( 15, 15) );
	dilate( imgThresh, imgThresh, element );
	erode( imgThresh, imgThresh, element );
	cvWaitKey(0);
	element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 5, 5),
                                       Point( 3, 3) );
	//erode( imgThresh, imgThresh, element )  ;
	//dilate( imgThresh, imgThresh, element ); 
	imshow("image1", imgThresh);
	imwrite("detectar_jugadores_eq1_4_" + nombre + ".png", imgThresh);
	cvWaitKey(0);
	findContours( imgThresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	printf("%i jugadores\n", contours.size());
	cvWaitKey(0);
	Mat img_contornos = img.clone();
	for	(int i = 0; i < contours.size(); i++) {
		drawContours( img_contornos, contours, i, Scalar(0,0,255), 5);
 	}
	imwrite("detectar_jugadores_eq1_5_" + nombre + ".png", img_contornos);
	imshow("image1", img_contornos);
	cvWaitKey(0);
	return contours;
}

// Equipo azul
vector<vector<Point> > detectarJugadores2 (Mat img, string nombre) {
	Mat imgHSV, imgThresh;
	vector<vector<Point> > contours;
	cvtColor(img, imgHSV, CV_BGR2HSV);
	inRange(imgHSV, Scalar(75,120,100), Scalar(130,256,256), imgThresh);
	imshow("image1", img);
	cvWaitKey(0);
	imshow("image1", imgHSV);
	imwrite("detectar_jugadores_eq2_1_" + nombre + ".png", imgHSV);
	cvWaitKey(0);
	imshow("image1", imgThresh);
	imwrite("detectar_jugadores_eq2_2_" + nombre + ".png", imgThresh);
	eliminarPaneles(imgThresh);
	imshow("image1", imgThresh);
	imwrite("detectar_jugadores_eq2_3_" + nombre + ".png", imgThresh);
	Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 40, 40 ),
                                       Point( 15, 15) );
	dilate( imgThresh, imgThresh, element );
	erode( imgThresh, imgThresh, element );
	cvWaitKey(0);
	element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 5, 5),
                                       Point( 3, 3) );
	erode( imgThresh, imgThresh, element )  ;
	dilate( imgThresh, imgThresh, element );
	imshow("image1", imgThresh);
	imwrite("detectar_jugadores_eq2_4_" + nombre + ".png", imgThresh);
	cvWaitKey(0);
	findContours( imgThresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	printf("%i jugadores\n", contours.size());
	cvWaitKey(0);
	Mat img_contornos = img.clone();
	for	(int i = 0; i < contours.size(); i++) {
		drawContours( img_contornos, contours, i, Scalar(255,0,0), 5);
 	}
	imwrite("detectar_jugadores_eq2_5_" + nombre + ".png", img_contornos);
	imshow("image1", img_contornos);
	cvWaitKey(0);
	return contours;
}

void detectarCesped (Mat img, string nombre) {
	Mat imgHSV, imgThresh;
	cvtColor(img, imgHSV, CV_BGR2HSV);
	inRange(imgHSV, Scalar(25,100,30), Scalar(75,256,256), imgThresh);
	eliminarPaneles(imgThresh);
	imshow("image1", imgThresh);
	imwrite("detectar_cesped1_" + nombre + ".png", imgThresh);
	cvWaitKey(0);
	Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 7, 7),
                                       Point( 4, 4) );
	erode( imgThresh, imgThresh, element );
	dilate( imgThresh, imgThresh, element );
	imshow("image1", imgThresh);
	imwrite("detectar_cesped2_" + nombre + ".png", imgThresh);
	cvWaitKey(0);
}

void detectarLineas (Mat img, string nombre) {
	Mat imgHSV, imgThresh, img_gray, img_canny;
	Mat img_aux = img.clone();
	vector<Vec4i> lineasP;
	// Mediante deteccion de color
	cvtColor(img, imgHSV, CV_BGR2HSV);
	// Mediante deteccion de color + hough
	inRange(imgHSV, Scalar(0,0,150), Scalar(255,100,255), imgThresh);
	eliminarPaneles(imgThresh);
	imshow("image1", imgThresh);
	imwrite("detectar_lineas_1_" + nombre + ".png", imgThresh);
	Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2, 2),
                                       Point( 1, 1) );
	dilate( imgThresh, imgThresh, element );
	erode( imgThresh, imgThresh, element );
	erode( imgThresh, imgThresh, element );
	dilate( imgThresh, imgThresh, element );
	HoughLinesP(imgThresh, lineasP, 1, CV_PI/180, 50,50,20);
	pintarLineasP(img, lineasP);
	imshow("image1", img);
	imwrite("detectar_lineas_2_" + nombre + ".png", img);
	cvWaitKey(0);
}

void detectarBalon (IplImage* img) {
	Mat imgHSV;
	int altura,anchura,anchura_fila,canales;
	uchar *data;
	int i,j;
	altura = img->height;
	anchura = img->width;
	anchura_fila = img->widthStep;
	canales = img->nChannels;
	data = (uchar *)img->imageData;
	cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("mainWin", 100, 100);
	for(i = 0; i < altura; i++) {
		for (j = 0; j < anchura; j++) {
			if (data[i*anchura_fila+j*canales + 2] > 210 &&
				data[i*anchura_fila+j*canales + 1] > 210 &&
				data[i*anchura_fila+j*canales + 0] < 60
				 ) {
				data[i*anchura_fila+j*canales + 0]=255;
				data[i*anchura_fila+j*canales + 1]=255;
				data[i*anchura_fila+j*canales + 2]=255;
			} else {
				data[i*anchura_fila+j*canales + 0]=0;
				data[i*anchura_fila+j*canales + 1]=0;
				data[i*anchura_fila+j*canales + 2]=0;
			}

		}
	}
	cvShowImage("mainWin", img);
	cvSaveImage("detectar_balon1_imagen4.png", img);
	cvWaitKey(0);
	cvtColor(img, imgHSV, CV_BGR2HSV);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size( 2, 2), Point( 1, 1));
	dilate(imgHSV, imgHSV, element);
	erode(imgHSV, imgHSV, element);
	erode(imgHSV, imgHSV, element);
	erode(imgHSV, imgHSV, element);
	erode(imgHSV, imgHSV, element);
	dilate(imgHSV, imgHSV, element);
	imshow("imagen",imgHSV);
	imwrite("detectar_balon2_imagen4.png", imgHSV);
	cvWaitKey(0);
	cvReleaseImage(&img);
}

void fueraJuegoA (Mat img, string nombre) {
	Mat img_channels[3];
	Mat img_gray;
	vector<Vec2f> lineas;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	Canny(img_gray, img_gray, 100, 200);
	eliminarPaneles(img_gray);
	HoughLines(img_gray, lineas, 1, CV_PI/180, 100);
	Mat img_aux = img.clone();
	pintarLineas(img_aux, lineas);
	imshow("image1", img_aux);
	imwrite("detectar_fuera_juego_A_1_" + nombre + ".png", img_aux);
	cvWaitKey(0);
	// Filtrado de rectas
	boolean rectaSimilar;
	// Eliminamos las horizontales
	float rho, theta;
	double a, b, x0, y0;
	for(size_t i = 0; i < lineas.size(); i++) {
		rho = lineas[i][0];
        theta = lineas[i][1];
        a = cos(theta);
		b = sin(theta);
        x0 = a * rho;
		y0 = b * rho;
        Point p1(cvRound(x0 + 1000 * (-b)),
                 cvRound(y0 + 1000 * (a)));
        Point p2(cvRound(x0 - 1000 * (-b)),
                 cvRound(y0 - 1000 * (a)));

		if (abs(p1.x - p2.x) > abs(p1.y - p2.y) || p2.x == p1.x) {
			lineas.erase(lineas.begin() + i);
			i--;
		} else {
			double pendiente_i = (p2.y - p1.y) / (p2.x - p1.x);
			for(size_t j = i+1; j < lineas.size(); j++) {
				rho = lineas[j][0];
			    theta = lineas[j][1];
				a = cos(theta);
				b = sin(theta);
				x0 = a * rho;
				y0 = b * rho;
				Point p1aux(cvRound(x0 + 1000 * (-b)),
						 cvRound(y0 + 1000 * (a)));
				Point p2aux(cvRound(x0 - 1000 * (-b)),
						 cvRound(y0 - 1000 * (a)));
				if (p2aux.x == p1aux.x) {
					lineas.erase(lineas.begin() + j);
					j--;
				} else {
					double distancia_x1 = abs(p1aux.x - p1.x);
					double distancia_y1 = abs(p1aux.y - p1.y);
					double distancia_x2 = abs(p2aux.x - p2.x);
					double distancia_y2 = abs(p2aux.y - p2.y);
					if (distancia_x1 + distancia_y1 + distancia_x2 + distancia_y2 < 100) {
						lineas.erase(lineas.begin() + j);
						j--;
					}
				}
			}
		}

	}
	rho = lineas[0][0];
	theta = lineas[0][1];
    a = cos(theta);
	b = sin(theta);
    x0 = a * rho;
	y0 = b * rho;
    Point p1a(cvRound(x0 + 1000 * (-b)),
             cvRound(y0 + 1000 * (a)));
    Point p2a(cvRound(x0 - 1000 * (-b)),
             cvRound(y0 - 1000 * (a)));
	printf("Recta 1: (%i, %i) (%i, %i)\n", p1a.x, p1a.y, p2a.x, p2a.y);
	rho = lineas[1][0];
	theta = lineas[1][1];
    a = cos(theta);
	b = sin(theta);
    x0 = a * rho;
	y0 = b * rho;
    Point p1b(cvRound(x0 + 1000 * (-b)),
             cvRound(y0 + 1000 * (a)));
    Point p2b(cvRound(x0 - 1000 * (-b)),
             cvRound(y0 - 1000 * (a)));
	printf("Recta 2: (%i, %i) (%i, %i)\n", p1b.x, p1b.y, p2b.x, p2b.y);
	pintarLineas(img, lineas);
	imshow("image1", img);
	imwrite("detectar_fuera_juego_A_2_" + nombre + ".png", img);
	cvWaitKey(0);
	
}

void fueraJuegoB (Mat img, int x, int y, string nombre) {
	vector<vector<Point> > contornos1 = detectarJugadores1(img, nombre);
	vector<vector<Point> > contornos2 = detectarJugadores2(img, nombre);

	Mat img_contornos = img.clone();
	for	(int i = 0; i < contornos1.size(); i++) {
		drawContours( img_contornos, contornos1, i, Scalar(0,0,255), 5);
		imshow("image1", img_contornos);
		cvWaitKey(0);
 	}

	for	(int i = 0; i < contornos2.size(); i++) {
		drawContours( img_contornos, contornos2, i, Scalar(255,0,0), 5);
		imshow("image1", img_contornos);
		cvWaitKey(0);
	}

	// Buscamos el jugador mas atrasado en cuanto a posición del ekipo defensor
	// (concretamente, su punto del cuerpo mas avanzado)
	Point p_def_max = Point(0,0);
	int xf_def_max = 0;
	for	(size_t i = 0; i < contornos1.size(); i++) {
		for (size_t j = 0; j < contornos1[i].size(); j++) {
			int yf_def = img_contornos.rows;
			int xf_def = (yf_def - y)*(contornos1[i][j].x - x)/(contornos1[i][j].y - y) + x;
			if (xf_def > xf_def_max) {
				p_def_max = contornos1[i][j];
				xf_def_max = xf_def;
			}
		}
	}
	int yf_def = img_contornos.rows;
	int xf_def = (yf_def - y)*(p_def_max.x - x)/(p_def_max.y - y) + x;
	line(img_contornos, Point(x, y), Point(xf_def, yf_def), Scalar(0,255,255), 3);
	imshow("image1", img_contornos);
	imwrite("detectar_fuera_juego_B_1_" + nombre + ".png", img_contornos);
	cvWaitKey(0);
	// Buscamos para cada jugador atacante su posición mas avanzada
	// y la comparamos con la anteriormente buscada
	for	(size_t i = 0; i < contornos2.size(); i++) {
		Point p_at_max = Point(0,0);
		for (size_t j = 0; j < contornos2[i].size(); j++) {
			if (contornos2[i][j].x > p_at_max.x) {
				p_at_max = contornos2[i][j];
			}
		}
		// Pintamos la línea en blanco si no está en fuera de juego
		int yf_atac = img_contornos.rows;
		int xf_atac = (yf_atac - y)*(p_at_max.x - x)/(p_at_max.y - y) + x;
		if (xf_atac < xf_def) {
			line(img_contornos, Point(x, y), Point(xf_atac, yf_atac), Scalar(255,255,255), 3);
		} else { // Pintamos la línea de negro si está en fuera de juego
			line(img_contornos, Point(x, y), Point(xf_atac, yf_atac), Scalar(0,0,0), 5);
		}
		imshow("image1", img_contornos);
		cvWaitKey(0);
	}
	imwrite("detectar_fuera_juego_B_2_" + nombre + ".png", img_contornos);
	cvWaitKey(0);
}

void fueraJuegoLineasVert (Mat img, string nombre) {
	vector<vector<Point> > contornos1 = detectarJugadores1(img, nombre);
	vector<vector<Point> > contornos2 = detectarJugadores2(img, nombre);

	Mat img_contornos = img.clone();
	for	(int i = 0; i < contornos1.size(); i++) {
		drawContours( img_contornos, contornos1, i, Scalar(0,0,255), 5);
		imshow("image1", img_contornos);
		cvWaitKey(0);
 	}

	for	(int i = 0; i < contornos2.size(); i++) {
		drawContours( img_contornos, contornos2, i, Scalar(255,0,0), 5);
		imshow("image1", img_contornos);
		cvWaitKey(0);
	}

	// Buscamos el jugador mas atrasado en cuanto a posición del ekipo defensor
	// (concretamente, su punto del cuerpo mas avanzado)
	Point p_def_max = Point(0,0);
	for	(size_t i = 0; i < contornos1.size(); i++) {
		for (size_t j = 0; j < contornos1[i].size(); j++) {
			if (contornos1[i][j].x > p_def_max.x) {
				p_def_max = contornos1[i][j];
			}
		}
	}
	line(img_contornos, Point(p_def_max.x, 0), Point(p_def_max.x, img_contornos.rows), Scalar(0,255,255), 5);
	imshow("image1", img_contornos);
	cvWaitKey(0);
	// Buscamos para cada jugador atacante su posición mas avanzada
	// y la comparamos con la anteriormente buscada
	for	(size_t i = 0; i < contornos2.size(); i++) {
		Point p_at_max = Point(0,0);
		for (size_t j = 0; j < contornos2[i].size(); j++) {
			if (contornos2[i][j].x > p_at_max.x) {
				p_at_max = contornos2[i][j];
			}
		}
		// Pintamos la línea en blanco si no está en fuera de juego
		if (p_at_max.x <= p_def_max.x) {
			line(img_contornos, Point(p_at_max.x, 0), Point(p_at_max.x, img_contornos.rows), Scalar(255,255,255), 5);
		} else { // Pintamos la línea de negro si está en fuera de juego
			line(img_contornos, Point(p_at_max.x, 0), Point(p_at_max.x, img_contornos.rows), Scalar(0,0,0), 5);
		}
		imshow("image1", img_contornos);
		cvWaitKey(0);
	}
	
}

void detectarTactica (Mat img, int umbral, string nombre) {
	vector<vector<Point> > contornos = detectarJugadores1(img, nombre);
	Mat img_contornos = img.clone();
	// Obtenemos el jugador menos adelantado del equipo (último defensor)
	int xmin = img.cols;
	int jugador;
	for	(size_t i = 0; i < contornos.size(); i++) {
		if (contornos[i][0].x < xmin) {
			jugador = i;
			xmin = contornos[i][1].x;
		}
	}
	// Eliminamos al jugador de la lista
	Point p_actual;
	Point p_antiguo = Point(contornos[jugador][0].x, contornos[jugador][0].y);
	int tono_rojo = 255;
	int jugadores_linea = 1;
	drawContours( img_contornos, contornos, jugador, Scalar(0,0,tono_rojo), 20);
	contornos.erase(contornos.begin() + jugador);
	
	printf("Estrategia: ");
	// Selección de línea táctica
	while (contornos.size() > 0) {
		// Obtenemos el jugador menos adelantado del equipo (último defensor)
		xmin = img.cols;
		for	(size_t i = 0; i < contornos.size(); i++) {	
			if (contornos[i][0].x < xmin) {
				jugador = i;
				xmin = contornos[i][1].x;
			}
		}
		// Cogemos al jugador de la lista
		Point p_actual = Point(contornos[jugador][0].x, contornos[jugador][0].y);

		if (p_actual.x - p_antiguo.x > umbral) { // Nueva línea
			printf("%i ", jugadores_linea);
			jugadores_linea = 1;
			tono_rojo -= 100;
			drawContours( img_contornos, contornos, jugador, Scalar(0,0,tono_rojo), 20);
			p_antiguo = p_actual;
		} else {
			jugadores_linea++;
			drawContours( img_contornos, contornos, jugador, Scalar(0,0,tono_rojo), 20);
		}
		imshow("image1", img_contornos);
		imwrite("detectar_tactica_" + nombre + ".png", img_contornos);
		cvWaitKey(0);
		contornos.erase(contornos.begin() + jugador);
	}

	printf("%i", jugadores_linea);
	cvWaitKey(0);
}


int main( int argc, char** argv ) {
	int mode = FUERA_JUEGO;
	Mat img;
	switch (mode) {
		case DETECCION_JUG1:
			img = imread("imagen1.jpg");
			detectarJugadores1(img, "imagen1");
			img = imread("imagen2.jpg");
			detectarJugadores1(img, "imagen2");
		break;
		case DETECCION_JUG2:
			img = imread("imagen1.jpg");
			detectarJugadores2(img, "imagen1");
			img = imread("imagen2.jpg");
			detectarJugadores2(img, "imagen2");
		break;
		case DETECCION_CESPED:
			img = imread("imagen1.jpg");
			detectarCesped(img, "imagen1");
			img = imread("imagen2.jpg");
			detectarCesped(img, "imagen2");
		break;
		case DETECCION_LINEAS:
			img = imread("imagen1.jpg");
			detectarLineas(img, "imagen1");
			img = imread("imagen2.jpg");
			detectarLineas(img, "imagen2");
		break;
		case DETECCION_BALON:
			img = imread("balon.jpg");
			detectarBalon(img);
		break;
		case FUERA_JUEGO:
			img = imread("imagen1.jpg");
			//fueraJuegoLineasVert(img, "imagen1");
			fueraJuegoA(img, "imagen1");
			fueraJuegoB(img, 625, -532, "imagen1");
		break;
		case TACTICA:
			img = imread("imagen3.jpg");
			detectarTactica(img, 150, "imagen3");
		break;
		case COLORMODEL_MODE:
			img = imread("2.jpg");
			cambiarModelosColor(img);
		break;
		case NORMHIST_MODE:
			img = imread("monedas.jpg");
			manipulacionContraste(img, "imagen1");
			img = imread("monedas.jpg");
			manipulacionContraste(img, "imagen2"); 
		break;
		case SEGMENTATION_MODE:
			img = imread("imagen2.jpg");
			//transformadaHough(img);
			img = imread("monedas.jpg");
			segmentacionUmbralizacion(img);
			segmentacionCrecimiento(img);
			//segmentacionMorph(img);
		break;
		case NOOPER_MODE:
		break;
	}
	return 0;
}