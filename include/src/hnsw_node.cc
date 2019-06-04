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

#include <vector>

#include "../n2/hnsw_node.h"

namespace n2 {

HnswNode::HnswNode(int id, const Data* data, int attributesNumber, const std::vector<std::string>& attr, int maxsize/*, int maxsize0*/)
: id_(id), data_(data), attributes_number_(attributesNumber), attributes_(attr), maxsize_(maxsize) {
}

void HnswNode::AddAttributesLevel(int attributes_id){
    attributes_id_.push_back(attributes_id);
    friends_at_attribute_id_[attributes_id].reserve(maxsize_+1);
}

// 预先分配0层以上每层的内存
void HnswNode::CopyHigherLevelLinksToOptIndex(char* mem_offset, long long memory_per_node_higher_level) const {
    // 1层的起始地址
    char* mem_data = mem_offset;
    std::vector<int> hash_;
    hash_.resize(attributes_id_.size()-1);
    for(int i=1;i<attributes_id_.size();i++){
        int dd = attributes_id_[i]%(attributes_id_.size()-1);
        while(hash_[dd]!=NULL){
            dd = (dd+1)%(attributes_id_.size()-1);
        }
        hash_[dd] = attributes_id_[i];
    }
    for (int i = 0; i < hash_.size(); ++i) {
        // 分配单层的内存
        CopyLinksToOptIndex(mem_data, hash_[i]);
        // 起始地址加上本层的内存大小（(friends_at_layer_[level].size()+1)*sizeof(int)）
        mem_data += memory_per_node_higher_level;
    }
}
// 预先分配0层的内存
void HnswNode::CopyDataAndLevel0LinksToOptIndex(char* mem_offset, int higher_level_offset, int maxsize) const {
    // 0层的起始地址
    char* mem_data = mem_offset;
    *((int*)(mem_data)) = higher_level_offset;
    mem_data += sizeof(int);
    // 分配单层的内存
    CopyLinksToOptIndex(mem_data, 0);
    mem_data += (sizeof(int) + sizeof(int)*maxsize);
    auto& data = data_->GetData();
    // 分陪存储该节点向量的内存
    for (size_t i = 0; i < data.size(); ++i) {
        *((float*)(mem_data)) = (float)data[i];
        mem_data += sizeof(float);
    }
}

void HnswNode::CopyLinksToOptIndex(char* mem_offset, int attributes_id) const {
    // 本层存储开始的起始地址
    char* mem_data = mem_offset;
    const auto& neighbors = friends_at_attribute_id_.find(attributes_id)->second;
    // 层数（除了0层）
    if(attributes_id!=0){
        *((int*)(mem_data)) = (int)(attributes_id_.size()-1);
        mem_data += sizeof(int);
        //邻居对应的属性id
        *((int*)(mem_data)) = (int)(attributes_id);
        // 本层的起始地址加上邻居的数量所需要的内存空间（sizeof(int)）即为下一步的起始地址
        mem_data += sizeof(int);
    }
    // mem_data指向的地址存储内容为邻居的数量
    *((int*)(mem_data)) = (int)(neighbors.size());
    // 本层的起始地址加上邻居的数量所需要的内存空间（sizeof(int)）即为下一步的起始地址
    mem_data += sizeof(int);
    // 从地址mem_data开始将每个邻居的id（int）一次存入
    for (size_t i = 0; i < neighbors.size(); ++i) {
        *((int*)(mem_data)) = (int)neighbors[i]->GetId();
        mem_data += sizeof(int);
    }
}

} // namespace n2
