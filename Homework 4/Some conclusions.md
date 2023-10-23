We've undertaken a challenging task with our homework, and we've organized our work into several notebooks for processing geo-data and training models.

practice4.ipynb: This serves as the primary notebook, housing essential code and comments to acquire datasets, initialize and train the model, and ultimately submit predictions.

processing-geo-dataset.ipynb: This notebook handles the processing of geo data and the extraction of embeddings. However, due to memory limitations, a novel approach was taken. The entire Russian dataset is too large to process in one go. To overcome this, we split the data into nine manageable chunks and processed them separately. Although the memory issue was mitigated, it resulted in the need to merge the processed chunks back together. In retrospect, optimizing memory usage and removing processed chunks might have been more efficient.

Practice4 both ru and uk datasets .ipynb: This notebook acts as a comprehensive wrapper for the preceding notebooks. It enabled the training of the model from practice4.ipynb on both the Russian and Ukrainian datasets. Due to the memory constraints, a practical solution was to utilize only the 8th and 9th chunks of the Russian dataset, merging them with the Ukrainian dataset. This approach, while offering some optimization, still led to long training times, approximately 3 hours per epoch.

In practice4.ipynb, we explored the use of both LSTM and GRU for the UniversalRNN class. Notably, the use of GRU proved to be more efficient. Training with GRU was slightly faster, taking approximately 50 minutes per epoch, compared to LSTM, which took around 56 minutes per epoch. Moreover, GRU yielded improved results, with an LB_GRU score of 0.49770, surpassing the LB_LSTM score of 0.44043.

Our description provides a clear overview of our work, the challenges we encountered, and the optimization strategies we employed. It showcases our problem-solving skills and demonstrates our dedication to achieving the best results.
