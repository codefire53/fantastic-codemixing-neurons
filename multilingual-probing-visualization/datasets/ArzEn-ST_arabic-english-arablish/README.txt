==========================================================================================================

	                                 ArzEn-ST Translations Corpus

                                                Release v1.0
                                                10 June 2023

===========================================================================================================
Summary
===========================================================================================================

ArzEn-ST translation corpus contains the translations of the ArzEn speech corpus transcriptions into monolingual Egyptian Arabic and monolingual English. ArzEn speech corpus is a 12-hours spontaneous conversational speech corpus, obtained through informal interviews held at the German University in Cairo. The participants discussed broad topics, including education, hobbies, work, and life experiences. The corpus contains significant amounts of code-switching, covering the four main types of code-switching: inter-sentential, intra-sentential, extra-sentential, and intra-word code-switching. The recordings were transcribed and translated into monolingual Egyptian Arabic and monolingual English. This corpus contains the transcriptions and translations.

Details on ArzEn Speech corpus, including information on participants, data collection, code-switching statistics, and transcription can be found in Hamed et al. (2020):
(1) Hamed, Injy, Ngoc Thang Vu, and Slim Abdennadher. "ArzEn: A speech corpus for code-switched Egyptian Arabic-English." In Proceedings of the 12th Language Resources and Evaluation Conference (2020).

Details on ArzEn-ST translation corpus, including translation guidelines and baseline Machine Translation and Speech Translation systems can be found in Hamed et al. (2022b):
(2) Hamed, Injy, Nizar Habash, Slim Abdennadher, and Ngoc Thang Vu. "ArzEn-ST: A Three-way Speech Translation Corpus for Code-Switched Egyptian Arabic - English." In Proceedings of the 7th Arabic Natural Language Processing Workshop (2022).

More information on the morphological code-switching present in the corpus can be found in Hamed et al. (2022a):
(3) Hamed, Injy, Pavel Denisov, Chia-Yu Li, Mohamed Elmahdy, Slim Abdennadher, and Ngoc Thang Vu. "Investigations on speech recognition systems for low-resource dialectal Arabic–English code-switching speech." In Computer Speech & Language Journal (2022).

===========================================================================================================
Description of Data
===========================================================================================================
README.txt  :   This file.

LICENSE.txt :   The license to use this corpus.

LREC2020_ArzEnSpeechCorpus.pdf :
                A paper describing the creation and use of ArzEn Speech corpus.

WANLP2022_ArzEnSTCorpus.pdf :
                A paper describing the creation and use of ArzEn-ST translation corpus.

data :	Contains the transcriptions and translations files. The data can be found in the three folders:
    transcriptions:	 		the original transcriptions obtained from the recordings.
    translations_EgyptianArabic: 	the translations of transcriptions into monolingual Egyptian Arabic.
    translations_English: 		the translations of transcriptions into monolingual English.
The corpus has predefined train, dev, and test splits, as outlined in Hamed et al. (2020). The corresponding train, dev, and test files are found in each of the three mentioned folders.

The data files contain one utterance per line. Each line has two columns (tab separated):
(1) First column: utterance ID, which consists of [speaker_ID]-[corpus_ID]-[Recording_ID]_[timestamp_start]-[timestamp_end]
(2) Second column: the original transcription utterance or its translation, depending on the file.

==========================================================================================================
References
==========================================================================================================
If you use this corpus, please cite the original ArzEn  papers for the transcriptions and translations:

Hamed, Injy, Ngoc Thang Vu, and Slim Abdennadher. "ArzEn: A speech corpus for code-switched Egyptian Arabic-English." In Proceedings of the 12th Language Resources and Evaluation Conference (2020).

Hamed, Injy, Nizar Habash, Slim Abdennadher, and Ngoc Thang Vu. "ArzEn-ST: A Three-way Speech Translation Corpus for Code-Switched Egyptian Arabic - English." In Proceedings of the 7th Arabic Natural Language Processing Workshop (2022).

@inproceedings{Hamed:2020:arzenspeechcorpus,
    title = "{ArzEn}: A speech corpus for code-switched {E}gyptian {A}rabic-{E}nglish",
    author = "Injy Hamed and Ngoc Thang Vu and Slim Abdennadher",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference (LREC)",
    year = "2020"
}

@inproceedings{Hamed:2022:arzenst,
    title = "{ArzEn-ST}: A Three-way Speech Translation Corpus for Code-Switched {E}gyptian {A}rabic - {E}nglish",
    author = "Injy Hamed and Nizar Habash and Slim Abdennadher and Ngoc Thang Vu",
    booktitle = "Proceedings of the 7th Arabic Natural Language Processing Workshop (WANLP)",
    year = "2022"
}

==========================================================================================================
