# Text-Summarizaton-on-MIND-News
**About Dataset**
==============================

The **MIND dataset** for news recommendation was collected from anonymized behavior logs of Microsoft News website. The data randomly sampled 1 million users who had at least 5 news clicks during 6 weeks from October 12 to November 22, 2019. To protect user privacy, each user is de-linked from the production system when securely hashed into an anonymized ID. Also collected the news click behaviors of these users in this period, which are formatted into impression logs. The impression logs have been used in the last week for test, and the logs in the fifth week for training. For samples in training set, used the click behaviors in the first four weeks to construct the news click history for user modeling. Among the training data, the samples in the last day of the fifth week used as validation set. This dataset is a small version of MIND (MIND-small), by randomly sampling 50,000 users and their behavior logs. Only training and validation sets are contained in the MIND-small dataset.

**news.tsv** contains the detailed information of news articles involved in the behaviors.tsv file. It has 7 columns, which are divided by the tab symbol:

* News ID
* Category
* SubCategory
* Title
* Abstract
* URL
* Title Entities (entities contained in the title of this news)
* Abstract Entities (entites contained in the abstract of this news)
