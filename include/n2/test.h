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

#pragma once

#include "spdlog/spdlog.h"

#include <queue>


namespace n2 {

class test {
public:
    test();
    test(int n);

    inline int addd();
    void setDistance(int h);
private:
    int distance_;
};

test::test(int n):distance_(n){

        }

        int test::addd(){
                return distance_;
        }

        void test::setDistance(int n){
                distance_ = n;
        }

} // namespace n2
