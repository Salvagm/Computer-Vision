#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/lexical_cast.hpp>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace std;

class Imagen{
	
	protected:

		Ptr<FeatureDetector> detector ;
		//Inizializamos el extractor de caracteristicas (descriptores)
		Ptr<DescriptorExtractor> extractor;

		vector< pair< vector<KeyPoint>,Mat> > parKeypointDescriptor;

		Mat imagen;

	public:
		
		//Coloca el puntero de el metodo para obtener los detectores
		void setDetector(const Ptr<FeatureDetector> &detector){			this->detector= detector;		}
		//devuelve el punto del metodo con el que obtenemos los detectores
		const Ptr<FeatureDetector> &getDetector(){			return detector;		}
		//Coloca el puntero con el metodo utilizado para extraer descriptores
		void setExtractor(const Ptr<DescriptorExtractor> &extractor ){	this->extractor= extractor;		}
		//devuelve el puntero del metodo utilizado para extraer descriptores
		const Ptr<DescriptorExtractor> &getExtractor(){			return extractor;		}
		//Coloca la imagen a color 
		void setImagenFinal(const Mat &imagen){			this->imagen=imagen;		}
		//Devuelve la imagen almacenada
		const Mat &getImagenFinal() {			return imagen;		} 		

		//Devuelve el vector compuesto por pares que contienen el vector de puntos junto con los descriptores asociados
		std::vector< pair<std::vector<KeyPoint> ,Mat> > &getPairKeypDesc(){
			return parKeypointDescriptor;
		}
		//Coloca un nuevo par de vectores de punto y descriptores dentro del vector.
		void setPairkeypDesc(const vector<KeyPoint> &keypoints, const Mat &descriptores){
			
			parKeypointDescriptor.push_back(make_pair(keypoints, descriptores));
			
		}
		//Comprueba si se ha utilizado los métodos de detección o extraccion de puntos al menos 1 vez
		bool isReady(){
			if(detector!=NULL && extractor!=NULL )
				return true;
			else
				return false;
		}
		// Sobrecarga del operador = para copiar una clase Imagen en otra
		Imagen &operator=(const Imagen &imagenCpy){
			
			if(this!=&imagenCpy){
				
				parKeypointDescriptor.clear();
				parKeypointDescriptor=imagenCpy.parKeypointDescriptor;
				
				detector=imagenCpy.detector;
				extractor=imagenCpy.extractor;
				imagen=imagenCpy.imagen;

				
			}

			return (*this);
		}
};

class Panoramica{

protected:
	// NEcesario para it_
	ros::NodeHandle nh_;
	//Imagen anterior y actual
	Imagen prevImg,currentImage;
	//Objeto encargado de recibir imagenes de ros
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	struct timespec ts1, ts2;

	float tiempoMatch, tiempoGetDescriptor;
	unsigned int numImagen, umbralDado, numMetodos;

	//vector que de pares de metodos
	vector< pair<string,string> > pairMethods;
public:

	//Recoge la imagen que se publica y la convierte a escala de grises
	Mat returnGrayImage(const sensor_msgs::ImageConstPtr& msg){
		cv_bridge::CvImagePtr cv_ptr;
	    try
	    {
	      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	    }
	    catch (cv_bridge::Exception& e)
	    {
	    	Mat matrixZero = Mat::zeros(0, 0, 0);
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return matrixZero;
	    }
	    // std::cerr<<" imagecb: "<<msg->header.frame_id<<" : "<<msg->header.seq<<" : "<<msg->header.stamp<<std::endl;
	    
		//src_gray->Almacena la imagen leida en escala de grises
	    Mat src_gray;
	    
		cvtColor( cv_ptr->image, src_gray, CV_BGR2GRAY );
		//Imagen a color patiendo de la escala de grises
		Mat imageColor;
	    cvtColor(src_gray, imageColor, CV_GRAY2BGR);

	    currentImage.setImagenFinal(imageColor);

		return src_gray;
	}
	//Extraemos los desriptores de la imagen segun el metodo de deteccion y extracion de caracteristicas
	void extractDescriptors(const string &detectorEntrada, const string &descriptorEntrada, const Mat &src_gray,Mat &descriptoresActuales, vector<KeyPoint> &keypoints){

	
		//Inicializamos los detectores
		Ptr<FeatureDetector> detector = FeatureDetector::create(detectorEntrada);
		//Inizializamos el extractor de caracteristicas (descriptores)
		Ptr<DescriptorExtractor> extractor= DescriptorExtractor::create(descriptorEntrada);

		//Obtenemos todos los puntos con shift
		detector->detect(src_gray,keypoints); 
		//Extraemos caracteristicas de la imagen
		extractor->compute(src_gray,keypoints, descriptoresActuales);
	
		if(descriptoresActuales.type()!=CV_32F){
			descriptoresActuales.convertTo(descriptoresActuales, CV_32F);
		}

		currentImage.setDetector(detector);
		currentImage.setExtractor(extractor);
		
		currentImage.setPairkeypDesc(keypoints,descriptoresActuales);
	}

	//Aplicamos un matching de puntos utilizando FLANN sobre los descriptores anteriores y actuales
	//Almacenamos el resultado en la variable good_matches
	void applyFlann(Mat &descriptoresAnteriores, Mat &descriptoresActuales,vector<DMatch> &good_matches){
		FlannBasedMatcher matcher;
		
		std::vector< DMatch > matches;

		matcher.match(descriptoresAnteriores,descriptoresActuales,matches);
		
		//Distancia minima la cual inicializamos a un valor alto
		//Posteriormente esta distancia se va decrementando
			double min_dist = 100;
		
		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < descriptoresAnteriores.rows; ++i )
		{ 
			double dist = matches[i].distance;
			if(dist < min_dist){
				min_dist= dist;
			}
		}
		
		for( int i = 0; i < descriptoresAnteriores.rows; i++ )
		{ 
			if( matches[i].distance < 1.8*min_dist && matches[i].distance<umbralDado )
				good_matches.push_back( matches[i]);
		}
		cout << "--FLANN--"<< endl;
		cout << "Matches descartados "<< matches.size()-good_matches.size()<< endl;
	}

	//Aplicamos un matching de puntos utilizando knn sobre los descriptores anteriores y actuales
	//Almacenamos el resultado en la variable good_matches
	void applyKnn(Mat &descriptoresAnteriores, Mat &descriptoresActuales, vector<DMatch> &good_matches){

		FlannBasedMatcher matcher;
		vector < vector<DMatch> > matchesKnn;
		
		matcher.knnMatch(descriptoresAnteriores,descriptoresActuales,matchesKnn ,2 );
		
		for (size_t i = 0; i < matchesKnn.size(); ++i)
		{ 
			if (matchesKnn[i].size() < 2)
				continue;

			const DMatch &m1 = matchesKnn[i][0];
			const DMatch &m2 = matchesKnn[i][1];

			if(m1.distance <= 0.8 * m2.distance && m1.distance<umbralDado)        
				good_matches.push_back(m1);     
		}
		
		cout << "--KNN--"<< endl;
		cout << "Matches descartados: "<< matchesKnn.size()-good_matches.size()<< endl;
	}


	void applyRansac(const std::vector<KeyPoint> &keypoints_ant,const Mat &H, const std::vector<DMatch> &good_matches, const Mat &src_gray, const std::vector<KeyPoint> &currentKeypoints, Mat &transformed_image  )
	{
		const std::vector<Point2f> points_ant_transformed(keypoints_ant.size());
		std::vector<Point2f> keypoints_ant_vector(keypoints_ant.size());
		cv::KeyPoint::convert(keypoints_ant,keypoints_ant_vector);

		//transformamos los puntos de la imagen anterior
		perspectiveTransform( keypoints_ant_vector, points_ant_transformed, H);

		//creamos una copia de la imagen actual que usaremos para dibujar
		cvtColor(src_gray, transformed_image, CV_GRAY2BGR);

		//los que esten mas lejos que este parametro se consideran outliers (o que la transformacion está mal calculada)
		//este valor es orientativo, podeis cambiarlo y ajustarlo a los valores
		float distance_threshold=10.0;
		int contdrawbuenos=0;
		int contdrawmalos=0;
		for ( int i =0;i<good_matches.size();i++)
		{
		    int ind        = good_matches.at(i).trainIdx ;
		    int ind_Ant    = good_matches.at(i).queryIdx;

		    cv::Point2f p=        currentKeypoints.at(ind).pt;
		    cv::Point2f p_ant=    points_ant_transformed[ind_Ant];

		    circle( transformed_image, p_ant, 5, Scalar(255,0,0), 2, 8, 0 ); //ant blue
		    circle( transformed_image, p, 5, Scalar(0,255,255), 2, 8, 0 ); //current yellow

		    Point pointdiff = p - points_ant_transformed[ind_Ant];
		        float distance_of_points=cv::sqrt(pointdiff.x*pointdiff.x + pointdiff.y*pointdiff.y);

		    if(distance_of_points < distance_threshold){ // los good matches se pintan con un circulo verde mas grand
		        contdrawbuenos++;
		        circle( transformed_image, p, 9, Scalar(0,255,0), 2, 8, 0 ); //current red
		    }
		    else{
		        contdrawmalos++;
		        line(transformed_image,p,p_ant,Scalar(0, 0, 255),1,CV_AA);
		    }
		}
		cout << "num correspondencias malas: "<< contdrawmalos<< endl;;
		cout << "num correspondencias buenas: "<< contdrawbuenos<< endl;
		imshow( "transformed", transformed_image );
	}

	//Metodo encargado de comprobar los puntos en los que coincide una imagen
	void imageCb(const sensor_msgs::ImageConstPtr& msg){
	    //src_gray->Almacena la imagen leida en escala de grises
	    Mat src_gray= returnGrayImage(msg);
	    //Variable para almacenar el resultado de ransca
	    Mat H;
	    //Si ha habido algun error terminamos la ejecución
	    if(src_gray.rows==0){return;}    	
	    
 		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		//Vector donde almacenamos los mejores matches
	    vector<DMatch> good_matches;
	    // Vector donde guardamos los keypoints de la imagen actual
	    vector <KeyPoint> currentkeypoints;
	    //Guardamos el par de Keypoint asociados a los Descriptores obtenido anteriormente
	    vector< pair< vector<KeyPoint>,Mat> > keyPDescAnteriores= prevImg.getPairKeypDesc();
		cout << "\n######################Iteraciones Nº"<< numImagen<<"######################"<< endl;
		for(unsigned int index=0; index<numMetodos; ++index){
			
			cout << "\n###################### ";
			cout << pairMethods[index].first<<"+"<< pairMethods[index].second <<" ";
			cout << "#########################"<< endl;
			//Guarda los resultados de los descriptores
			Mat descriptoresActuales;
			
			currentkeypoints.clear();
			clock_gettime( CLOCK_REALTIME, &ts1 );
		  	extractDescriptors(pairMethods[index].first,pairMethods[index].second,src_gray, descriptoresActuales, currentkeypoints);
			clock_gettime( CLOCK_REALTIME, &ts2 );
			tiempoGetDescriptor=(float) ( 1.0*(1.0*ts2.tv_nsec - ts1.tv_nsec*1.0)*1e-9+ 1.0*ts2.tv_sec - 1.0*ts1.tv_sec );
			cout << "Extracion de caracteristicas: "<<tiempoGetDescriptor<< endl;
			
			if(prevImg.isReady()){
				
				Mat img_matches, img_transform;
				
				good_matches.clear();

				clock_gettime( CLOCK_REALTIME, &ts1 );
				// POR defecto aplicamos KNN, si queremos aplicar FLANN comentamos una linea
				// y descomentamos la otra

				// applyFlann(keyPDescAnteriores[index].second, descriptoresActuales,good_matches) ;
				 applyKnn(keyPDescAnteriores[index].second,descriptoresActuales, good_matches);
				 
				clock_gettime( CLOCK_REALTIME, &ts2 );

				tiempoMatch=(float) ( 1.0*(1.0*ts2.tv_nsec - ts1.tv_nsec*1.0)*1e-9+ 1.0*ts2.tv_sec - 1.0*ts1.tv_sec );
				cout << "Tiempo en calculo de Matches:  "<<tiempoMatch<< endl;
		 		
		 		//Pinta solo lo puntos emparejados
		 	  	drawMatches( prevImg.getImagenFinal(), keyPDescAnteriores[index].first, currentImage.getImagenFinal(), currentImage.getPairKeypDesc()[index].first,good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		 		
		 	  	//FINDHOMOGRAPHY (RANSCA)
		 	  	obj.clear();
		 	  	scene.clear();

				for( int i = 0; i < good_matches.size(); ++i )
				{
					//-- Get the keypoints from the good matches
					obj.push_back( keyPDescAnteriores[index].first[ good_matches[i].queryIdx ].pt );
					scene.push_back( currentImage.getPairKeypDesc()[index].first[ good_matches[i].trainIdx ].pt );
				}
				//RANSAC solo funciona con una cantidad mayor o igual 4 de puntos 
				if(obj.size()>4 && scene.size()>4){
					Mat H = findHomography( obj, scene, CV_RANSAC );
					applyRansac(keyPDescAnteriores[index].first,H,good_matches, src_gray,currentkeypoints, img_transform);
				string s="/home/salva/ImagenesPanorama/imagen"+boost::lexical_cast<string>(numImagen)+boost::lexical_cast<string>(index)+pairMethods[index].first+pairMethods[index].second+".png";
					imwrite(s, img_transform );	
				}
			   //FINDHOMOGRAPHY (RANSAC)
				

				imshow("matches",img_matches);
				//imwrite(const string &filename, InputArray img)
			}
		}	    
		
		++numImagen;
		prevImg=currentImage;
		currentImage.getPairKeypDesc().clear();
	    cv::waitKey(3);

	}

  Panoramica() : it_(nh_){
  	numImagen=0;
  	umbralDado=150;
  	tiempoMatch=0;

    // Subscrive to input video feed and publish output video feed
    // Cuando llegue una imagen al topico, llama a la funcion imgeCB
    image_sub_ = it_.subscribe("/camera/rgb/image_color", 1, &Panoramica::imageCb, this);
    
	pairMethods.push_back(make_pair("SIFT", "SIFT"));
	pairMethods.push_back(make_pair("SURF", "SURF"));
	pairMethods.push_back(make_pair("SIFT", "SURF"));
	pairMethods.push_back(make_pair("SURF", "SIFT"));
	
	pairMethods.push_back(make_pair("ORB", "SIFT"));
	pairMethods.push_back(make_pair("ORB", "SURF"));
	pairMethods.push_back(make_pair("MSER", "SIFT"));
	pairMethods.push_back(make_pair("MSER", "SURF"));
	
	pairMethods.push_back(make_pair("ORB","ORB"));
	pairMethods.push_back(make_pair("MSER", "BRIEF"));
	pairMethods.push_back(make_pair("MSER","ORB"));
    
    numMetodos= pairMethods.size();
  }


};


int main(int argc, char **argv) {
	
	// if(std::strcmp("ALL", argv[1])==0)
	cv::initModule_nonfree();
		
	ros::init(argc, argv, "panoramica"); // Inicializa un nuevo nodo llamado panoramica
	Panoramica p;


	ros::spin();
	return 0;
}