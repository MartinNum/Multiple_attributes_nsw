// Copyright 2017 Kakao Corp. <http://www.kakaocorp.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>
#include <random>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <fstream> 
#include <iostream> 
#include <algorithm>
#include <time.h>
#include <sys/time.h>  
#include <stdio.h> 

#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

// #include "../../include/n2/hnsw.h"


using namespace std;    


void SplitString(const string& s, vector<float>& v, const string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    
    while(string::npos != pos2)
    {
        v.push_back(atof(s.substr(pos1, pos2-pos1).c_str()));
         
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(atof(s.substr(pos1).c_str()));
}

 int64_t getCurrentTime()      //直接调用这个函数就行了，返回值最好是int64_t，long long应该也可以
    {    
       struct timeval tv;    
       gettimeofday(&tv,NULL);    //该函数在sys/time.h头文件中
       return tv.tv_sec * 1000 + tv.tv_usec / 1000;    
    }    


int main(int argc, char** argv)
{
    // std::vector<std::string> attributes = {"浙江","高校","其他"};
    // std::vector<int> id = {0,3,6};
    // std::map<int,std::vector<std::string>> node_attributes_;
    // node_attributes_[0].push_back("null");
    // int n = 1;
    // if(attributes.size() == 3){
    //     int attributesNumber = attributes.size();
    //     for(int i=1;i<1<<attributesNumber;++i){
    //         for(int j=0;j<attributesNumber;++j){
    //             if(i>>j&0x1){
    //                 node_attributes_[n].push_back(attributes[j]);
    //             }
    //         }
    //         ++n;
    //     }
    // }
    // for(int i=0;i<node_attributes_.size();++i){
    //         cout<<node_attributes_[i].size()<<endl;
    //         for(int j=0;j<node_attributes_[i].size();++j){
    //             cout<<node_attributes_[i][j]<<endl;
    //         }
    //     }

    // node_attributes_.clear();
    // for(int i=0;i<id.size();++i){
    //     node_attributes_[id[i]].reserve(2);
    // }


    // std::vector<std::string> &kkk = attributes;

    // std::set<std::string> all_attributes_;

    // for(int i=0;i<attributes.size();++i){
    //     all_attributes_.insert(attributes[i]);
    // }

    // std::map<int,std::vector<std::string>> id_attribute_;
    // std::map<int,std::vector<std::string>> node_attributes_;
    // id_attribute_[0].push_back("null");
    // id_attribute_[1].push_back("浙江");
    // id_attribute_[2].push_back("高校");
    // id_attribute_[3].push_back("浙江");
    // id_attribute_[3].push_back("高校");
    // id_attribute_[4].push_back("其他");
    // id_attribute_[5].push_back("浙江");
    // id_attribute_[5].push_back("其他");
    // id_attribute_[6].push_back("高校");
    // id_attribute_[6].push_back("其他");
    // id_attribute_[7].push_back("浙江");
    // id_attribute_[7].push_back("高校");
    // id_attribute_[7].push_back("其他");

    // node_attributes_[0].push_back("null");
    // node_attributes_[1].push_back("浙江");
    // node_attributes_[2].push_back("高校");
    // node_attributes_[3].push_back("浙江");
    // node_attributes_[3].push_back("高校");
    // node_attributes_[4].push_back("其他");
    // node_attributes_[5].push_back("垃圾");
    // node_attributes_[5].push_back("其他");
    // node_attributes_[6].push_back("高校");
    // node_attributes_[6].push_back("其他");
    // node_attributes_[7].push_back("浙江");
    // node_attributes_[7].push_back("高校");
    // node_attributes_[7].push_back("其他");
    // int h = id_attribute_.size();
    // for(int ni=0;ni<node_attributes_.size();++ni){
    // int levl;
    // int nn = false;
    // for(int ai=0;ai<h;++ai){
    //     if(node_attributes_[ni].size()==id_attribute_[ai].size()){
    //         int flog = 0;
    //         for(int nj=0;nj<node_attributes_[ni].size();++nj){
    //             for(int aj=0;aj<id_attribute_[ai].size();++aj){
    //                 if(node_attributes_[ni][nj]==id_attribute_[ai][aj]){
    //                     ++flog;
    //                     break;
    //                 }
    //             }
    //         }
    //         if(flog==node_attributes_[ni].size()){
    //             levl = ai;
    //             nn = true;
    //             break;
    //         }
    //     }
    // }
    // if(nn==true){
    //     cout<<"已存在："<<levl<<endl;
    //     // qnode->AddAttributesLevel(levl);
    // }else{
    //     cout<<"不存在："<<ni<<endl;
    //     int newl = id_attribute_.size();
    //     for(int mm=0;mm<node_attributes_[ni].size();++mm){
    //         id_attribute_[newl].push_back(node_attributes_[ni][mm]);
    //         // qnode->AddAttributesLevel(id_attribute_.size());
    //     }
    // }
    // }


    // cout<<id_attribute_.size();

    // int g = (rand() % (9-1))+ 1;
    // for(int i=0;i<1000;i++){
    //     int g = (rand() % (9-0+1))+ 0;
    // }

    // bool flog = true;
    // cout<<flog<<endl;
    // cout<<flog<<endl;

    // vector<int> a = {0,1,2,5,7,20,4,2};
    // vector<int> b;
    // b.resize(a.size()-1);
    // for(int i=1;i<a.size();i++){
    //     int dd = a[i]%(a.size()-1);
    //     while(b[dd]!=NULL){
    //         dd = (dd+1)%(a.size()-1);
    //     }
    //     b[dd] = a[i];
    // }

    // for(int a:b){
    //     cout<< a <<endl;
    // }

    // string a = "马丁程";
    // cout<<sizeof(string)<<endl;
    // }

    // cout<<sizeof(string)<<endl;

    // int a = 45;
    // string b = "madingcheng";

    // char* model_ = new char[1000];
    // char* ptr = model_;

    // *((int*)(ptr)) = a;

    // ptr += sizeof(int);

    // *((string*)(ptr)) = b;

    // int c = *((int*)(model_));

    // string d = *((string*)(model_+sizeof(int)));

    // cout<<c<<endl;

    // cout<<d<<endl;

    // int ef = 3;

    // int k = 10;
    
    // cout<<3*0.1*10/10<<endl;


    // ifstream myfile("/home/martin/桌面/sift_ma/sift_base.txt"); 
    // string temp; 
    // if (!myfile.is_open()) 
    // { 
    //     cout << "未成功打开文件" << endl; 
    // } 

    // while(getline(myfile,temp)){
    //     vector<float> v;
    //     SplitString(temp, v," "); //可按多个字符来分隔;
    //     if(v.size() == 131){
    //         for(int i = 0; i != v.size()-3; ++i)
    //         cout << v[i] <<" ";
    //     cout<<endl;
    //     }
    // }
    // cout << endl;


    
    // myfile.close(); 
    // return 0; 

    // vector<int> h = {1,2,3};
    // h.pop_back();
    // for(int g: h){
    //     cout<<g<<" ";
    // }

    // time_t tt = time(NULL);
    // cout<<tt<<endl;
    // time_t tt1 = time(NULL);
    // cout<<tt1<<endl;
    // cout<<tt1-tt<<endl;

    int64_t a = getCurrentTime();
    int64_t b = getCurrentTime();

    std::cout<<"nowTime: "<<a-b<<"\n";
    


}
