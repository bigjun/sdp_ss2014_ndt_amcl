/**
* 2D NDT-MCL Node. 
* This application runs the ndt-mcl localization based on a map and laser scanner and odometry. 
* 
* The initialization is based now on ground truth pose information (should be replaced with manual input). 
* 
* The visualization depends on mrpt-gui 
* 
* More details about the algorithm:
* Jari Saarinen, Henrik Andreasson, Todor Stoyanov and Achim J. Lilienthal, Normal Distributions Transform Monte-Carlo Localization (NDT-MCL)
* IEEE/RSJ International Conference on Intelligent Robots and Systems November 3-8, 2013, Tokyo Big Sight, Japan
* 
* @author Jari Saarinen (jari.p.saarinen@gmail.com)
* 
*	@TODO Initialization from GUI
* @TODO Global initialization possibility 
* Known issues: in launch file (or hard coded parameters) you have to set the same resolution for NDT map as is saved -- otherwise it wont work
*/

// Enable / Disable visualization
#define USE_VISUALIZATION_DEBUG
 
#include <mrpt/utils/CTicTac.h>

#ifdef USE_VISUALIZATION_DEBUG
	#include <mrpt/gui.h>	
	#include <mrpt/base.h>
	#include <mrpt/opengl.h>
	#include <GL/gl.h>
	#include "ndt_mcl/CMyEllipsoid.h"
#endif

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf/transform_listener.h>
#include <boost/foreach.hpp>
#include <sensor_msgs/LaserScan.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/Pose.h"

#include "ndt_mcl/impl/ndt_mcl.hpp"
#include <ndt_map/ndt_map.h>


#ifdef USE_VISUALIZATION_DEBUG
    //here is a bunch of visualization code based on the MRPT's GUI components
    #include "ndt_mcl/impl/mcl_visualization.hpp"
#endif

class NdtMclNode
{
public:
    NdtMclNode();
    void callback(const sensor_msgs::LaserScan::ConstPtr& scan);
    void initialPoseReceived(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg);
    void sendMapToRviz(lslgeneric::NDTMap<pcl::PointXYZ> &map);
    bool sendROSOdoMessage(Eigen::Vector3d mean,Eigen::Matrix3d cov, ros::Time ts);

    static bool hasSensorOffsetSet;
    static bool isFirstLoad;

private:
    Eigen::Affine3d getAsAffine(float x, float y, float yaw);

    ros::NodeHandle nodeHandle;
    ros::NodeHandle paramHandle;

    ros::Publisher mcl_pub;
    ros::Publisher ndtmap_pub; 
    ros::Subscriber initial_pose_sub_;
    ros::Subscriber scan_sub;

    bool userInitialPose;
    bool hasNewInitialPose;

    /// Initial pose stuff
    double ipos_x;
    double ipos_y;
    double ipos_yaw;
    double ivar_x;
    double ivar_y;
    double ivar_yaw;

    NDTMCL<pcl::PointXYZ> *ndtmcl;

    ///Laser sensor offset
    float offx;
    float offy;
    float offa;

    Eigen::Affine3d Told;
    Eigen::Affine3d Todo;

    mrpt::utils::CTicTac TT;
    std::string tf_odo;
    std::string tf_state;
    std::string tf_laser_link;
    std::string tf_world;

    ros::Duration tf_timestamp_tolerance;
    double tf_tolerance;
};

bool NdtMclNode::hasSensorOffsetSet = false;
bool NdtMclNode::isFirstLoad = true;

NdtMclNode::NdtMclNode():
    paramHandle("~")
{
    this->hasNewInitialPose = false;

    this->ivar_x = 0.0;
    this->ivar_y = 0.0;
    this->ivar_yaw = 0.0;

    #ifdef USE_VISUALIZATION_DEBUG	
        initializeScene();
    #endif

	TT.Tic();

	std::string input_laser_topic; 
	this->paramHandle.param<std::string>("input_laser_topic", input_laser_topic, std::string("/base_scan"));

	this->paramHandle.param<std::string>("tf_base_link", this->tf_state, std::string("/base_link"));
	this->paramHandle.param<std::string>("tf_laser_link", this->tf_laser_link, std::string("/hokuyo1_link"));
	this->paramHandle.param<std::string>("tf_odom", this->tf_odo, std::string("odom"));
	this->paramHandle.param<std::string>("tf_world", this->tf_world, std::string("map"));

	bool use_sensor_pose;
	this->paramHandle.param<bool>("use_sensor_pose", use_sensor_pose, false);

	double sensor_pose_x, sensor_pose_y, sensor_pose_th;
	this->paramHandle.param<double>("sensor_pose_x", sensor_pose_x, 0.);
	this->paramHandle.param<double>("sensor_pose_y", sensor_pose_y, 0.);
	this->paramHandle.param<double>("sensor_pose_th", sensor_pose_th, 0.);

    //flag to indicate that we want to load a map
	bool loadMap = false;
	std::string mapName;

    //indicates if we want to save the map in a regular intervals
	bool saveMap = true;
	std::string output_map_name;

	this->paramHandle.param<bool>("load_map_from_file", loadMap, false);
	this->paramHandle.param<std::string>("map_file_name", mapName, std::string("basement.ndmap"));
	this->paramHandle.param<bool>("save_output_map", saveMap, true);
	this->paramHandle.param<std::string>("output_map_file_name", output_map_name, std::string("ndt_mapper_output.ndmap"));

	bool forceSIR = false;
	this->paramHandle.param<bool>("forceSIR", forceSIR, false);
		
	this->paramHandle.param<bool>("set_initial_pose", this->userInitialPose, false);
	this->paramHandle.param<double>("initial_pose_x", this->ipos_x, 0.);
	this->paramHandle.param<double>("initial_pose_y", this->ipos_y, 0.);
	this->paramHandle.param<double>("initial_pose_yaw", this->ipos_yaw, 0.);

    this->paramHandle.param<double>("tf_timestamp_tolerance", this->tf_tolerance, 1.0);
    tf_timestamp_tolerance.fromSec(tf_tolerance);

	if(this->userInitialPose == true)
        this->hasNewInitialPose = true;

	// Prepare the map
    double resolution = 0.4;
    this->paramHandle.param<double>("map_resolution", resolution , 0.2);
	fprintf(stderr,"USING RESOLUTION %lf\n",resolution);
	lslgeneric::NDTMap<pcl::PointXYZ> ndmap(new lslgeneric::LazyGrid<pcl::PointXYZ>(resolution));
	ndmap.setMapSize(80.0, 80.0, 1.0);

	if(loadMap)
    {
		fprintf(stderr,"Loading Map from '%s'\n",mapName.c_str());
		ndmap.loadFromJFF(mapName.c_str());
	}

	ndtmcl = new NDTMCL<pcl::PointXYZ>(resolution,ndmap,-0.5);
	if(forceSIR)
        ndtmcl->forceSIR = true;

	fprintf(stderr,"*** FORCE SIR = %d****", forceSIR);

	//Set up our output
	this->mcl_pub = this->nodeHandle.advertise<nav_msgs::Odometry>("ndt_mcl", 10);

	//Set the subscribers and setup callbacks and message filters
    this->scan_sub = this->nodeHandle.subscribe(input_laser_topic, 1, &NdtMclNode::callback, this);
	this->ndtmap_pub = this->nodeHandle.advertise<visualization_msgs::MarkerArray>( "NDTMAP", 0);
	this->initial_pose_sub_ = this->nodeHandle.subscribe("initialpose", 1, &NdtMclNode::initialPoseReceived, this);

	this->offx = sensor_pose_x;
	this->offy = sensor_pose_y;
    this->offa = sensor_pose_th;
	NdtMclNode::hasSensorOffsetSet = true;

	fprintf(stderr,"Sensor Pose = (%lf %lf %lf)\n",offx, offy, offa);	
}

/**
 * Callback for laser scan messages 
 */
void NdtMclNode::callback(const sensor_msgs::LaserScan::ConstPtr& scan)
{
    static int counter = 0;
    counter++;

    static tf::TransformListener tf_listener;
    double looptime = this->TT.Tac();
    this->TT.Tic();
    fprintf(stderr,"Lt( %.1lfms %.1lfHz seq:%d) -",looptime*1000,1.0/looptime,scan->header.seq);
    
    if(NdtMclNode::hasSensorOffsetSet == false)
        return;
    double gx,gy,gyaw,x,y,yaw;

    //Get state information
    tf::StampedTransform transform;
    tf_listener.waitForTransform(this->tf_world, this->tf_state, scan->header.stamp,ros::Duration(1.0));

    //Ground truth --- Not generally available so should be changed to the manual initialization
    try
    {
        tf_listener.lookupTransform(this->tf_world, this->tf_state, scan->header.stamp, transform);
        gyaw = tf::getYaw(transform.getRotation());  
        gx = transform.getOrigin().x();
        gy = transform.getOrigin().y();
    }
    catch (tf::TransformException ex)
    {
        gyaw = 0.0;
        gx = 0.0;
        gy = 0.0;
        ROS_ERROR("%s",ex.what());
        //return;
    }

    //Odometry 
    try
        {
            tf_listener.lookupTransform(this->tf_world, this->tf_state, scan->header.stamp, transform);
            yaw = tf::getYaw(transform.getRotation());  
            x = transform.getOrigin().x();
            y = transform.getOrigin().y();
    }
    catch (tf::TransformException ex)
    {
        yaw = 0.0;
        x = 0.0;
        y = 0.0;
        ROS_ERROR("%s",ex.what());
        //return;
    }

    mrpt::utils::CTicTac tictac;
    tictac.Tic();
    
    //Number of scans
    int N =(scan->angle_max - scan->angle_min)/scan->angle_increment;

    //Pose conversions
    Eigen::Affine3d T = this->getAsAffine(x,y,yaw);
    Eigen::Affine3d Tgt = this->getAsAffine(gx,gy,gyaw);

    if(userInitialPose && hasNewInitialPose)
    {
        gx = ipos_x;
        gy = ipos_y;
        gyaw = ipos_yaw;
    }

    if(NdtMclNode::isFirstLoad || hasNewInitialPose)
    {
        fprintf(stderr,"Initializing to (%lf, %lf, %lf)\n",gx,gy,gyaw);
        /// Initialize the particle filter
        ndtmcl->initializeFilter(gx, gy,gyaw,0.2, 0.2, 2.0*M_PI/180.0, 150);
        this->Told = T;
        this->Todo = Tgt;
        hasNewInitialPose = false;
    }

    //Calculate the differential motion from the last frame
    Eigen::Affine3d Tmotion = this->Told.inverse() * T;

    //Just integrates odometry for the visualization
    this->Todo = this->Todo * Tmotion;

    if(NdtMclNode::isFirstLoad==false)
    {
        if( (Tmotion.translation().norm()<0.005 && fabs(Tmotion.rotation().eulerAngles(0,1,2)[2])<(0.2*M_PI/180.0)))
        {
            Eigen::Vector3d dm = this->ndtmcl->getMean();
            Eigen::Matrix3d cov = this->ndtmcl->pf.getDistributionVariances();
            this->sendROSOdoMessage(dm, cov, scan->header.stamp + this->tf_timestamp_tolerance);
            double Time = tictac.Tac();
            fprintf(stderr,"Time elapsed %.1lfms\n",Time*1000);
            return;
        }
    }
    
    this->Told = T;
    
    //Calculate the laser pose with respect to the base
    float dy =offy;
    float dx = offx;
    float alpha = atan2(dy,dx);
    float L = sqrt(dx*dx+dy*dy);
    
    //Laser pose in base frame
    float lpx = L * cos(alpha);
    float lpy = L * sin(alpha);
    float lpa = offa;
    
    //Laser scan to PointCloud expressed in the base frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);	
    for (int j=0;j<N;j++)
    {
        double r  = scan->ranges[j];
        if(r>=scan->range_min && r<scan->range_max && r>0.3 && r<20.0)
        {
            double a  = scan->angle_min + j*scan->angle_increment;
            pcl::PointXYZ pt;
            pt.x = r*cos(a+lpa)+lpx;
            pt.y = r*sin(a+lpa)+lpy;
            pt.z = 0.1+0.02 * (double)rand()/(double)RAND_MAX;
            cloud->push_back(pt);
        }
    }

    // Now we have the sensor origin and pointcloud -- Lets do MCL
    if( (Tmotion.translation().norm()>0.01 || fabs(Tmotion.rotation().eulerAngles(0,1,2)[2])>(0.5*M_PI/180.0)))
        //predicts, updates and resamples if necessary (ndt_mcl.hpp)
        this->ndtmcl->updateAndPredict(Tmotion, *cloud);

    //Maximum aposteriori pose
    Eigen::Vector3d dm = this->ndtmcl->getMean();
    Eigen::Matrix3d cov = this->ndtmcl->pf.getDistributionVariances();

    double Time = tictac.Tac();
    fprintf(stderr,"Time elapsed %.1lfms (%lf %lf %lf) \n",Time*1000,dm[0],dm[1],dm[2]);

    //Spit out the pose estimate
    this->sendROSOdoMessage(dm, cov, scan->header.stamp + this->tf_timestamp_tolerance);
    NdtMclNode::isFirstLoad = false;

    if(counter%50==0)
        sendMapToRviz(ndtmcl->map);

    ///This is all for visualization
    #ifdef USE_VISUALIZATION_DEBUG
        Eigen::Vector3d origin(dm[0] + L * cos(dm[2]+alpha),dm[1] + L * sin(dm[2]+alpha),0.1);
        Eigen::Affine3d ppos = this->getAsAffine(dm[0],dm[1],dm[2]);
        
        lslgeneric::transformPointCloudInPlace(ppos, *cloud);
        mrpt::opengl::COpenGLScenePtr &scene = win3D.get3DSceneAndLock();	
        win3D.setCameraPointingToPoint(gx,gy,1);

        if(counter%2000==0)
            gl_points->clear();

        scene->clear();
        scene->insert(plane);
        
        addMap2Scene(ndtmcl->map, origin, scene);
        addPoseCovariance(dm[0],dm[1],cov,scene);
        addScanToScene(scene, origin, cloud);
        addParticlesToWorld(ndtmcl->pf,Tgt.translation(),dm, Todo.translation());
        scene->insert(gl_points);
        scene->insert(gl_particles);
        win3D.unlockAccess3DScene();
        win3D.repaint();

        if (win3D.keyHit())
        {
            mrpt::gui::mrptKeyModifier kmods;
            int key = win3D.getPushedKey(&kmods);
        }
    #endif
}

/**
 *  RVIZ NDT-MAP Visualization stuff
 */
void NdtMclNode::sendMapToRviz(lslgeneric::NDTMap<pcl::PointXYZ> &map)
{
    std::vector<lslgeneric::NDTCell<pcl::PointXYZ>*> ndts;
    ndts = map.getAllCells();
    fprintf(stderr,"SENDING MARKER ARRAY MESSAGE (%lu components)\n",ndts.size());
    visualization_msgs::MarkerArray marray;

    for(unsigned int i=0;i<ndts.size();i++)
    {
        Eigen::Matrix3d cov = ndts[i]->getCov();
        Eigen::Vector3d m = ndts[i]->getMean();            
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> Sol (cov);
        Eigen::Matrix3d evecs;
        Eigen::Vector3d evals;

        evecs = Sol.eigenvectors().real();
        evals = Sol.eigenvalues().real();
        
        Eigen::Quaternion<double> q(evecs);
    
        visualization_msgs::Marker marker;
        marker.header.frame_id = "odom";
        marker.header.stamp = ros::Time();
        marker.ns = "NDT";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = m[0];
        marker.pose.position.y = m[1];
        marker.pose.position.z = m[2];
        
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();
        
        marker.scale.x = 100.0*evals(0);
        marker.scale.y = 100.0*evals(1);
        marker.scale.z = 100.0*evals(2);
        
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        
        marray.markers.push_back(marker);
    }
    
    this->ndtmap_pub.publish(marray);
    
    for(unsigned int i=0;i<ndts.size();i++){
        delete ndts[i];
    }
}

bool NdtMclNode::sendROSOdoMessage(Eigen::Vector3d mean,Eigen::Matrix3d cov, ros::Time ts)
{
    static tf::TransformBroadcaster br;
    if(!NdtMclNode::isFirstLoad)
    {
        //*************************************
        // Code copied from the amcl node;
        // subtracts base->odom from map->base
        //*************************************
        tf::Stamped<tf::Pose> odom_to_map;
        static tf::TransformListener tf_listener;
        try
        {
                tf::Transform tmp_tf(tf::createQuaternionFromYaw(mean[2]),tf::Vector3(mean[0],mean[1],0.0));
                tf::Stamped<tf::Pose> tmp_tf_stamped (tmp_tf.inverse(), ts-tf_timestamp_tolerance, this->tf_state);
                tf_listener.transformPose(this->tf_odo, tmp_tf_stamped, odom_to_map);
        }
        catch(tf::TransformException ex)
        {
                ROS_ERROR("Failed to subtract base to odom transform: %s", ex.what());
                return false;
        }

        tf::Transform latest_tf_ = tf::Transform(tf::Quaternion(odom_to_map.getRotation()), tf::Point(odom_to_map.getOrigin()));
        tf::StampedTransform tmp_tf_stamped(latest_tf_.inverse(), ts, tf_world, this->tf_odo);

        //*****************************************************
        // Creates a pose message and publishes the transforms
        //*****************************************************
        nav_msgs::Odometry odom_message;

        static int seq = 0;
        odom_message.header.stamp = ts;
        odom_message.header.seq = seq;
        odom_message.header.frame_id = this->tf_world;
        odom_message.child_frame_id = this->tf_odo;

        odom_message.pose.pose.position.x = tmp_tf_stamped.getOrigin().x();
        odom_message.pose.pose.position.y = tmp_tf_stamped.getOrigin().y();
        tf::Quaternion q;
        //q.setRPY(0,0,tmp_tf_stamped.getRotation().yaw());
        odom_message.pose.pose.orientation.x = tmp_tf_stamped.getRotation().x();
        odom_message.pose.pose.orientation.y = tmp_tf_stamped.getRotation().y();
        odom_message.pose.pose.orientation.z = tmp_tf_stamped.getRotation().z();
        odom_message.pose.pose.orientation.w = tmp_tf_stamped.getRotation().w();

        odom_message.pose.covariance[0] = cov(0,0);
        odom_message.pose.covariance[1] = cov(0,1);
        odom_message.pose.covariance[2] = 0;
        odom_message.pose.covariance[3] = 0;
        odom_message.pose.covariance[4] = 0;
        odom_message.pose.covariance[5] = 0;

        odom_message.pose.covariance[6] = cov(1,0);
        odom_message.pose.covariance[7] = cov(1,1);
        odom_message.pose.covariance[8] = 0;
        odom_message.pose.covariance[9] = 0;
        odom_message.pose.covariance[10] = 0;
        odom_message.pose.covariance[11] = 0;

        odom_message.pose.covariance[12] = 0;
        odom_message.pose.covariance[13] = 0;
        odom_message.pose.covariance[14] = 0;
        odom_message.pose.covariance[15] = 0;
        odom_message.pose.covariance[16] = 0;
        odom_message.pose.covariance[17] = 0;

        odom_message.pose.covariance[18] = 0;
        odom_message.pose.covariance[19] = 0;
        odom_message.pose.covariance[20] = 0;
        odom_message.pose.covariance[21] = 0;
        odom_message.pose.covariance[22] = 0;
        odom_message.pose.covariance[23] = 0;

        odom_message.pose.covariance[24] = 0;
        odom_message.pose.covariance[25] = 0;
        odom_message.pose.covariance[26] = 0;
        odom_message.pose.covariance[27] = 0;
        odom_message.pose.covariance[28] = 0;
        odom_message.pose.covariance[29] = 0;

        odom_message.pose.covariance[30] = 0;
        odom_message.pose.covariance[31] = 0;
        odom_message.pose.covariance[32] = 0;
        odom_message.pose.covariance[33] = 0;
        odom_message.pose.covariance[34] = 0;
        odom_message.pose.covariance[35] = cov(2,2);

        seq++;
        this->mcl_pub.publish(odom_message);
        br.sendTransform(tmp_tf_stamped);
    }
    else
    {
        tf::Transform transform;
        transform.setOrigin( tf::Vector3(0.0,0.0,0.0) );  
        tf::Quaternion q;
        q.setRPY(0,0,0);
        transform.setRotation( q );
        br.sendTransform(tf::StampedTransform(transform, ts, this->tf_world, this->tf_odo));
    }

    return true;
}

void NdtMclNode::initialPoseReceived(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
{
    tf::Pose ipose;
    tf::poseMsgToTF(msg->pose.pose, ipose);
    this->ipos_x = ipose.getOrigin().x();
    this->ipos_y = ipose.getOrigin().y();

    double pitch, roll;
    ipose.getBasis().getEulerYPR(this->ipos_yaw, pitch, roll);

    this->ivar_x = msg->pose.covariance[0];
    this->ivar_x = msg->pose.covariance[6];
    this->ivar_x = msg->pose.covariance[35];

    this->hasNewInitialPose = true;
}

/**
 * Convert x,y,yaw to Eigen::Affine3d 
 */ 
Eigen::Affine3d NdtMclNode::getAsAffine(float x, float y, float yaw)
{
    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())
            * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Translation3d v(x,y,0);
    Eigen::Affine3d T = v*m;
    return T;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	ros::init(argc, argv, "NDT-MCL");

    NdtMclNode ndtMcl;
	ros::spin();
	
	return 0;
}
