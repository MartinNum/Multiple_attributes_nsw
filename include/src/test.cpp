#include <iostream>
#include "../n2/test.h"
 
using namespace std;

namespace n2 {
        test::test(int n):distance_(n){

        }

        int test::addd(){
                return distance_;
        }

        void test::setDistance(int n){
                distance_ = n;
        }
}

