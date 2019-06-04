// random_device rd;
    // mt19937 mt(rd());
    // normal_distribution<double> dist(0.0, 1.0);

    // const int F = 3;
    // n2::Hnsw index(F, "L2");
    // for(int i=0; i < 1000; ++i){
    //     vector<float> v(F);
    //     generate(v.begin(), v.end(), [&mt, &dist] { return dist(mt); });
    //     index.AddData(v);
    // }
    // vector<pair<string, string>>
    //     configs = {{"M", "5"}, {"MaxM0", "10"}, {"NumThread", "4"}};
    // index.SetConfigs(configs);
    // index.Fit();
    // index.SaveModel("test.n2");

    // n2::Hnsw otherway(F, "L2");
    // for(int i=0; i < 1000; ++i){
    //     vector<float> v(F);
    //     generate(v.begin(), v.end(), [&mt, &dist] { return dist(mt); });
    //     otherway.AddData(v);
    // }
    // otherway.Build(5, 10, -1, 4);
    // n2::Hnsw index2;
    // index2.LoadModel("test.n2");
    // int search_id = 1,
    //     k = 3;
    // vector<pair<int, float>> result;
    // index2.SearchById(search_id, k, -1, result);
    // cout << "[SearchById]: K-NN for " << search_id << " ";
    // for(auto ret : result){
    //     cout << "(" << ret.first << "," << ret.second << ") ";
    // }
    // cout << endl;

    // vector<float> v(F);
    // generate(v.begin(), v.end(), [&mt, &dist] { return dist(mt); });
    // index2.SearchByVector(v, k, -1, result);
    // cout << "[SearchByVector]: K-NN for [";
    // for(auto e : v){
    //     cout << e << ",";
    // }
    // cout << "] ";
    // for(auto ret : result){
    //     cout << "(" << ret.first << "," << ret.second << ") ";
    // }
    // cout << endl;