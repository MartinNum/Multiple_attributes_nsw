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
#include <iostream>
#include <algorithm>
#include <time.h>
#include <sys/time.h>  
#include <stdio.h> 

#include "../n2/hnsw.h"
// #include "../../include/n2/test.h"



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
    random_device rd;
    mt19937 mt(rd());
    normal_distribution<double> dist(0.0, 1.0);
    const int F = 128;
    // n2::Hnsw index(F, "L2");
    // // for(int i=0; i < 1000; ++i){
    // //     vector<float> v(F);
    // //     generate(v.begin(), v.end(), [&mt, &dist] { return dist(mt); });
    // //     index.AddData(v);
    // // }




    // ifstream myfile("/home/martin/桌面/sift_ma/sift_base.txt"); 
    ifstream myquire("/home/martin/桌面/sift_ma/sift_query.txt"); 
    string temp; 
    // if (!myfile.is_open()) 
    // { 
    //     cout << "未成功打开文件" << endl; 
    // } 

    // while(getline(myfile,temp)){
    //     vector<float> v;
    //     SplitString(temp, v," "); //可按多个字符来分隔;
    //     if(v.size() == 131){
    //         for(int i = 0; i <3; ++i)
    //             v.pop_back();
    //         index.AddData(v);
    //     }
    // }
    // cout << endl;
    // myfile.close(); 


    vector<vector<float>> inquire;
    while(getline(myquire,temp)){
        vector<float> v;
        SplitString(temp, v," "); //可按多个字符来分隔;
        if(v.size() == 131){
            for(int i = 0; i <3; ++i)
                v.pop_back();
            inquire.push_back(v);
        }
    }
    cout << endl;
    myquire.close(); 



    // vector<pair<string, string>> configs = {{"M", "15"}, {"MaxM0", "10"}, {"NumThread", "4"}};
    // index.SetConfigs(configs);
    // index.Fit();
    // index.SaveModel("test.n2");
    
    // n2::Hnsw otherway(F, "L2");
    // for(int i=0; i < 100; ++i){
    //     vector<float> v(F);
    //     generate(v.begin(), v.end(), [&mt, &dist] { return dist(mt); });
    //     otherway.AddData(v);
    // }
    // otherway.Build(5, 10, -1, 4);
    n2::Hnsw index2;
    index2.LoadModel("test.n2");
    while(true){
        int search_id, searchByVectorId, k = 100;
        cin>>search_id;
        cout<<"查询数据集的大小 = "<<inquire.size()<<endl;
        cin>>searchByVectorId;
        vector<pair<int, float>> result_id;
        index2.SearchById(search_id, k, -1, result_id);
        cout << "[SearchById]: K-NN for " << search_id << " ";
        for(auto ret : result_id){
            cout << "(" << ret.first << "," << ret.second << ") ";
        }
        cout << endl;

        vector<pair<int, float>> result_vector;

        vector<float> v(F);
        generate(v.begin(), v.end(), [&mt, &dist] { return dist(mt); });
        std::vector<std::string> attributes = {"null"};
        // vector<float> kk = {0.55557,-1.89095,-0.515863};

        int64_t t1 = getCurrentTime();
        index2.SearchByVector_new(inquire[searchByVectorId], attributes, k, -1, result_vector);
        int64_t t2 = getCurrentTime();
        std::cout<<"time of SearchByVector_new = "<<t2-t1<<endl;

        cout << "[SearchByVector]: K-NN for [";
        for(auto e : v){
            cout << e << ",";
        }
        cout << "] ";
        for(auto ret : result_vector){
            cout << "(" << ret.first << "," << ret.second << ") ";
        }

        vector<pair<int, float>> result_vector_violence;

        index2.SearchByVector_new_violence(inquire[searchByVectorId], attributes, k, -1, result_vector_violence);
        cout << "[SearchByVectorViolence]: K-NN for [";
        for(auto e : v){
            cout << e << ",";
        }
        cout << "] ";
        for(auto ret : result_vector_violence){
            cout << "(" << ret.first << "," << ret.second << ") ";
        }
        cout << endl;

        int hg = 0;
        for(auto ret : result_vector){
            for(auto retv : result_vector_violence){
                if(ret.first == retv.first){
                    hg++;
                    break;
                }
            }
        }

        cout<<"精度为： "<<(hg*1.0/k)*100<<"%"<<endl;

        }
}
