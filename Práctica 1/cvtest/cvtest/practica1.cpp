
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;

const int OP_RESOLUCIO = 0;
const int OP_COLOR = 1;
const int OP_UMBRALITZACIO = 2;
const int OP_RENOU = 3;
const int OP_HISTOGRAMA = 4;
const int OP_CONTORNS = 5;
const int OP_SEGMENTACIO = 6;
const int OP_CARACTERISTIQUES = 7;


/******************************************************************************************************************/
/********************************************** FUNCIONS AUXIL·LIARS **********************************************/
/******************************************************************************************************************/

Mat separarCanal(Mat img, int x, int y, int tipus) {
	Mat canal(img.rows, img.cols, tipus);
	Mat img_out[1] = {canal};
	int from_to[] = {x, y};
	mixChannels(&img, 1, img_out, 1, from_to, 1);
	return img_out[0];
}


void filtroGaussiano(Mat img, string tipusRenou) {
	Mat imgAux;
	GaussianBlur(img, imgAux, Size(25,25), 1, 1);
	imwrite("ferrari_renou" + tipusRenou + "_gaussianblur_1.png", imgAux);
	GaussianBlur(img, imgAux, Size(25,25), 3, 3);
	imwrite("ferrari_renou" + tipusRenou + "_gaussianblur_3.png", imgAux);
	GaussianBlur(img, imgAux, Size(25,25), 5, 5);
	imwrite("ferrari_renou" + tipusRenou + "_gaussianblur_5.png", imgAux);
	GaussianBlur(img, imgAux, Size(25,25), 7, 7);
	imwrite("ferrari_renou" + tipusRenou + "_gaussianblur_7.png", imgAux);
}


void filtroMediana(Mat img, string tipusRenou) {
	Mat imgAux;
	medianBlur(img, imgAux, 1);
	imwrite("ferrari_renou" + tipusRenou + "_medianblur_1.png", imgAux);
	medianBlur(img, imgAux, 3);
	imwrite("ferrari_renou" + tipusRenou + "_medianblur_3.png", imgAux);
	medianBlur(img, imgAux, 5);
	imwrite("ferrari_renou" + tipusRenou + "_medianblur_5.png", imgAux);
	medianBlur(img, imgAux, 7);
	imwrite("ferrari_renou" + tipusRenou + "_medianblur_7.png", imgAux);
}

/******************************************************************************************************************/



/*
 * Modificar la resolución espacial y la cuantificación. Comparar resultados calidad visual versus tamaño fichero.
 */
void resolucio_espacial_i_quantificacio(Mat imatge) {

	int const nivellsGris[4] = {256, 64, 16, 2};
	double escales[4];
	escales[0] = 1; //cas particular sense canvis
	for (int i=1; i<4; i++) {
		escales[i] = (double) (nivellsGris[i]) / 256.0;
	}

	int const tamanys_n[4] = {1024, 512, 256, 64};
	int const tamanys_m[4] = {768,  384, 192, 48};

	//imatge original a gris
	cvtColor(imatge, imatge, CV_BGR2GRAY);

	//per cada nivell de gris
	for (int i=0; i<4; i++) {
		Mat imgAux;
		int aux = 0;

		// transformació resolució en amplitud
		imatge.convertTo(imgAux, 8, escales[i]);
		imgAux.convertTo(imgAux, 8, 1.0/escales[i]);

		//per cada tamany
		for (int j=0; j<4; j++) {

			// transformació resulució espacial
			if (j == 0) {
				aux = 0;
			} else {
				aux = j - 1;
			}
			float escala = (float) tamanys_n[j] / tamanys_n[aux];
			resize(imgAux, imgAux, Size(), escala, escala, INTER_NEAREST);

			string res_n = std::to_string(static_cast<long long>(tamanys_n[j]));
			string res_m = std::to_string(static_cast<long long>(tamanys_m[j]));
			string res_d = std::to_string(static_cast<long long>(nivellsGris[i]));
			string nomArxiu = string("ferrari_") + res_n + "x" + res_m + "_" + res_d + ".png";

			// guardam imatge
			imwrite(nomArxiu, imgAux);
		}
	}
}


/*
 * Sobre una imagen en color de la escena, cambiar de modelo de color RGB a HSL.
 * Seleccionar la banda que obtenga más información sobre la intensidad de la imagen en cada modelo.
 * Hacer cambios de modelos entre las imágenes.
 */
void color(Mat imatge_RGB) {
     
	// CONVERSIONS

	// Conversió RGB a HSL
	Mat imatge_HSL;
	cvtColor(imatge_RGB, imatge_HSL, CV_BGR2HLS);
	// Conversió RGB a CIE Lab
	Mat imatge_CIE;
	cvtColor(imatge_RGB, imatge_CIE, CV_BGR2XYZ);
	// Conversió RGB a HSV (HSI)
	Mat imatge_HSV;
	cvtColor(imatge_RGB, imatge_HSV, CV_BGR2HSV);


	// SEPARACIÓ CANALS
	Mat imgAux;
	// RGB
	imwrite("path/ferrari_RGB.jpg", imatge_RGB);
	imgAux = separarCanal(imatge_RGB, 2, 2, CV_8UC3); //Red
	imwrite("path/ferrari_RGB_R.jpg", imgAux);
	imgAux = separarCanal(imatge_RGB, 1, 1, CV_8UC3); //Green
	imwrite("path/ferrari_RGB_G.jpg", imgAux);
	imgAux = separarCanal(imatge_RGB, 0, 0, CV_8UC3); //Blue
	imwrite("path/ferrari_RGB_B.jpg", imgAux);
	// HSL
	imwrite("path/ferrari_HSL.jpg", imatge_HSL);
	imgAux = separarCanal(imatge_HSL, 2, 0, CV_8UC1); //Hue
	imwrite("path/ferrari_HSL_H.jpg", imgAux);
	imgAux = separarCanal(imatge_HSL, 1, 0, CV_8UC1); //Saturation
	imwrite("path/ferrari_HSL_S.jpg", imgAux);
	imgAux = separarCanal(imatge_HSL, 0, 0, CV_8UC1); //Lightness
	imwrite("path/ferrari_HSL_L.jpg", imgAux);
	// CIE Lab
	imwrite("path/ferrari_CIE.jpg", imatge_CIE);
	imgAux = separarCanal(imatge_CIE, 2, 0, CV_8UC1); //Lluminància
	imwrite("path/ferrari_CIE_C.jpg", imgAux);
	imgAux = separarCanal(imatge_CIE, 1, 0, CV_8UC1); //Compnent a (verd --> vermell)
	imwrite("path/ferrari_CIE_I.jpg", imgAux);
	imgAux = separarCanal(imatge_CIE, 0, 0, CV_8UC1); //Compnent b (blau --> groc)
	imwrite("path/ferrari_CIE_E.jpg", imgAux);
	// HSV (HSI)
	imwrite("path/ferrari_HSV.jpg", imatge_HSV);
	imgAux = separarCanal(imatge_HSV, 2, 0, CV_8UC1); //Hue
	imwrite("path/ferrari_HSV_H.jpg", imgAux);
	imgAux = separarCanal(imatge_HSV, 1, 0, CV_8UC1); //Saturation
	imwrite("path/ferrari_HSV_S.jpg", imgAux);
	imgAux = separarCanal(imatge_HSV, 0, 0, CV_8UC1); //Value
	imwrite("path/ferrari_HSV_V.jpg", imgAux);
}


/*
 * Abrir una imagen o secuencia de imágenes y realizar una binarización basada en thresholding o umbralizacion básica con el objetivo de separar el fondo de la imagen de los objetos.
 */
void umbralitzacio(Mat imatge) {

	// binarització threesholding
	Mat imgAux1, imgAux2;
	cvtColor(imatge, imgAux1, CV_BGR2GRAY);

	int const valorsThresh[4] = {100, 150, 200, 240};
	for (int i=0; i<4; i++) {
		threshold(imgAux1, imgAux2, valorsThresh[i], 255, THRESH_BINARY);

		// guardam imatge
		string val = std::to_string(static_cast<long long>(valorsThresh[i]));
		string nomArxiu = string("poma_threshold_") + val + ".jpg";
		imwrite(nomArxiu, imgAux2);
	}


}


/*
 * Aplicación de funciones para mejora y eliminación de ruido.
 * Perturbar la imagen original con diferentes tipos de ruido y aplicar algoritmos de eliminación del mismo.
 * Justificar el mejor método en cada caso.
 */
void renou(Mat imatge) {

	// AFEGIM RENOU
	Mat canalsImatge[3];
	split(imatge, canalsImatge);

	// Renou gaussià
	Mat imgRenouGaussia;
	Mat imatge_gaussian_noise = canalsImatge[0].clone();
	Mat media = canalsImatge[0].clone();
	media.setTo(100);
	for (int i = 0; i < imatge.channels(); i++) {
		randn(imatge_gaussian_noise, 128, 30);
		canalsImatge[i] = (imatge_gaussian_noise - media) + canalsImatge[i];
		canalsImatge[i] = (media - imatge_gaussian_noise) + canalsImatge[i];
	}
	merge(canalsImatge, 3, imgRenouGaussia);
	imwrite("ferrari_renou_gaussia_generat.png", imgRenouGaussia);
	cvWaitKey(0);
	
	// Renou salt&pepper
	Mat imgRenouSaltPepper;
	Mat imatge_saltpepper_noise = Mat::zeros(imatge.rows, imatge.cols, CV_8U);
	randu(imatge_saltpepper_noise, 0, 255);
	Mat black = imatge_saltpepper_noise < 12;
	Mat white = imatge_saltpepper_noise > 253;
	media.setTo(128);
	for (int i = 0; i < imatge.channels(); i++) {
		canalsImatge[i].setTo(255, white);
		canalsImatge[i].setTo(0, black);
	}
	merge(canalsImatge, 3, imgRenouSaltPepper);
	imwrite("ferrari_renou_saltpepper_generat.png", imgRenouSaltPepper);
	cvWaitKey(0);


	// ELIMINAM RENOU

	// Filtro gaussiano
	filtroGaussiano(imgRenouGaussia, "gaussia");
	filtroGaussiano(imgRenouSaltPepper, "saltpepper");

	// Filtro mediana
	filtroMediana(imgRenouGaussia, "gaussia");
	filtroMediana(imgRenouSaltPepper, "saltpepper");

}


/*
 * Realizar operaciones de mejora del contraste de la imagen mediante manipulación del histograma.
 * Indicar para que sirve cada una.
 */
void histograma(Mat imatge) {
	Mat imatgeEqualitzada;

	// separam els canals RGB, els equalitzam i els tornam a juntar
	Mat canalsImatge[3];
	split(imatge, canalsImatge);
	for (int i=0; i<imatge.channels(); i++) {
		equalizeHist(canalsImatge[i], canalsImatge[i]);
	}
	merge(canalsImatge, imatge.channels(), imatgeEqualitzada);

	/// guardam imatge
	imwrite("ferrari_equalitzat.jpg", imatgeEqualitzada);
}


/*
 * Realizar la detección de contornos mediante filtros básicos.
 * Realizar técnicas de realce de contornos.
 */
void contorns(Mat imatge) {
	Mat imgAux;
	cvtColor(imatge, imgAux, CV_BGR2GRAY);

	// OPERADOR SOBEL
	Mat imgSobel;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(imgAux, grad_x, CV_16S, 1, 0, 3);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(imgAux, grad_y, CV_16S, 0, 1, 3);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgSobel);
	imwrite("ferrari_sobel.jpg", imgSobel);

	// OPERADOR LAPLACE
	Mat imgLaplace;
	Laplacian(imatge, imgLaplace, 8);
	imwrite("ferrari_laplace.jpg", imgLaplace);

	// OPERADOR CANNY
	Mat imgCanny;
	Canny(imatge, imgCanny, 210, 230);
	imwrite("ferrari_canny.jpg", imgCanny);


	// realçar contorns
	Mat imatgeSuavitzada;
	medianBlur(imatge, imatgeSuavitzada, 3);
	for (int k=1; k<=7; k=k+2) {
		imgAux = imatge + k * (imatge - imatgeSuavitzada);
		string cnst = std::to_string(static_cast<long long>(k));
		imwrite("ferrari_realcecontornos_" + cnst+ ".jpg", imgAux);
	}
}


/*
 * Realizar la segmentación de una imagen de niveles de gris.
 * Aplicar métodos de umbralización, de crecimiento de regiones y morfológicos.
 */
void segmentacio() {
	Mat imgCanicas, imgAux;
	for (int i=0; i<5; i++) {
		// llegim imatge
		string str_i = std::to_string(static_cast<long long>(i+1));
		string nom_arxiu = "canicas" + str_i;
		imgCanicas = imread(nom_arxiu + ".jpg");

		
		// SEGMENTACIÓ: Creixement de regions
		cvtColor(imgCanicas, imgAux, CV_BGR2GRAY);
		Canny(imgAux, imgAux, 120, 130);
		int n = 500;
		unsigned char colorActual = 100;
		Mat mask = Mat::zeros(imgCanicas.rows - 2, imgCanicas.cols - 2, CV_8UC1);
		for (int i=0; i < n; i++) {
			int x = rand() % mask.cols;
			int y = rand() % mask.rows;
			Point p(x,y);
			if (mascara.at<uchar>(p) == 0) {
				floodFill(mask, imgAux, p, colorActual, 0, Scalar(), Scalar(), 4);
				char valorI[2];
				itoa(i, valorI, 10);
				imwrite("segmentacion_crecimientoregiones_" + nom_arxiu + ".png", mask);
				colorActual += 100;
			}
		}


		// SEGMENTACIÓ: Umbralització adaptativa
		cvtColor(imgCanicas, imgAux, CV_BGR2GRAY);
		adaptiveThreshold(imgAux, imgAux, 150, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 111, 15);
		imwrite("segmentacion_umbralizacion_" + nom_arxiu + ".png", imgAux);
		

		// OPERACIONS MORFOLÒGIQUES
		equalizeHist(imgAux, imgAux);
		Mat elementEstr = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(2, 2));
		erode(imgAux, imgAux, elementEstr);
		dilate(imgAux, imgAux, elementEstr);
		imwrite("segmentacion_operacionesmorfologicas_" + nom_arxiu + ".png", imgAux);
	}
}


/*
 * Realizar una extracción de características para la selección de regiones.
 * Mirar propiedades métricas, topológicas y de textura.
 */
void extraccio_caracteristiques(Mat imatge) {
	// S'ha usat ImageTool
}




int main()
{
    Mat imatge = imread("ferrari.jpg");

	if(!imatge.data) { // error
		printf("No existeix la imatge.\n");
		return -1;
	}

	int operacio = OP_RESOLUCIO; // ={OP_RESOLUCIO, OP_COLOR, OP_UMBRALITZACIO, OP_RENOU, OP_HISTOGRAMA, OP_CONTORNS, OP_SEGMENTACIO, OP_CARACTERISTIQUES}

	switch (operacio) {
		case OP_RESOLUCIO:
			resolucio_espacial_i_quantificacio(imatge);
		break;
		case OP_COLOR:
			color(imatge);
		break;
		case OP_UMBRALITZACIO:
			umbralitzacio(imatge);
		break;
		case OP_RENOU:
			renou(imatge);
		break;
		case OP_HISTOGRAMA:
			histograma(imatge);
		break;
		case OP_CONTORNS:
			contorns(imatge);
		break;
		case OP_SEGMENTACIO:
			segmentacio();
		break;
		case OP_CARACTERISTIQUES:
			extraccio_caracteristiques(imatge);
		break;
		default:
			return -1; // error
		break;
	}

	waitKey(0);

	return 0;
}