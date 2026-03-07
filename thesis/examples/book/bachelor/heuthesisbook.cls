%%
%% This is file `examples/book/bachelor/heuthesisbook.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% heuthesis.dtx  (with options: `bookcls')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2021-2026 by LIWenhui <lwh@hrbeu.edu.cn>
%% 
%% This file may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3a
%% of this license or (at your option) any later version.
%% The latest version of this license is in:
%% 
%% http://www.latex-project.org/lppl.txt
%% 
%% To produce the documentation run the original source files ending with `.dtx'
%% through LaTeX.
%% 
\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{heuthesisbook}
[2021/05/06 3.0.15 Harbin Engineering University Thesis Template]
\RequirePackage{ifthen}
\RequirePackage{kvoptions}
\SetupKeyvalOptions{
  family=heu,
  prefix=heu@,
  setkeys=\kvsetkeys}
\newif\ifheu@bachelor
\newif\ifheu@master
\newif\ifheu@doctor
\newif\ifheu@postdoc
\define@key{heu}{type}{%
  \heu@bachelorfalse
  \heu@masterfalse
  \heu@doctorfalse
  \heu@postdocfalse
  \expandafter\csname heu@#1true\endcsname}
\newif\ifheu@productdevelopment
\newif\ifheu@projectdplanning
\newif\ifheu@projectdesign
\newif\ifheu@applicationresearch
\define@key{heu}{researchtype}{%
  \heu@productdevelopmentfalse
  \heu@projectdplanningfalse
  \heu@projectdesignfalse
  \heu@applicationresearchfalse
  \expandafter\csname heu@#1true\endcsname}
\ifheu@productdevelopment\relax\else
\ifheu@projectdplanning\relax\else
\ifheu@projectdesign\relax\else
\heu@applicationresearchtrue
\fi
\fi
\fi
\newif\ifheu@qingdao
\newif\ifheu@yantai
\newif\ifheu@harbin
\define@key{heu}{campus}{%
  \heu@qingdaofalse
  \heu@yantaifalse
  \heu@harbinfalse
  \expandafter\csname heu@#1true\endcsname}
\ifheu@yantai\relax\else
\ifheu@qingdao\relax\else
\heu@harbintrue
\fi
\fi
\newif\ifheu@english
\newif\ifheu@chinese
\define@key{heu}{language}{%
  \heu@englishfalse
  \heu@chinesefalse
  \expandafter\csname heu@#1true\endcsname}
\ifheu@english\relax\else
\heu@chinesetrue
\fi
\newif\ifheu@geometrynewone
\newif\ifheu@geometrynewtwo
\newif\ifheu@geometrynewno
\define@key{heu}{newgeometry}{%
  \heu@geometrynewonefalse
  \heu@geometrynewtwofalse
  \heu@geometrynewnofalse
  \expandafter\csname heu@geometrynew#1true\endcsname}
\newif\ifheu@zijvregu
\newif\ifheu@zijvword
\define@key{heu}{zijv}{%
  \heu@zijvregufalse
  \heu@zijvwordfalse
  \expandafter\csname heu@zijv#1true\endcsname}
\newif\ifheu@citetwocomma
\newif\ifheu@citetwoendash
\heu@citetwoendashtrue
\define@key{heu}{citetwo}{%
  \heu@citetwocommafalse
  \heu@citetwoendashfalse
  \expandafter\csname heu@citetwo#1true\endcsname
}
\DeclareBoolOption[false]{arialtoc}
\DeclareBoolOption[false]{arialtitle}
\DeclareBoolOption[true]{raggedbottom}
\DeclareBoolOption[true]{pifootnote}
\DeclareBoolOption[false]{glue}
\DeclareBoolOption[false]{tocfour}
\DeclareBoolOption[true]{chapterhang}
\DeclareBoolOption[false]{profdegree}
\DeclareBoolOption[false]{subtitle}
\DeclareBoolOption[false]{debug}
\DeclareBoolOption[false]{openright}
\DeclareBoolOption[false]{pageempty}
\DeclareBoolOption[false]{library}
\DeclareBoolOption[false]{capcenterlast}
\DeclareBoolOption[false]{subcapcenterlast}
\DeclareBoolOption[false]{absupper}
\DeclareBoolOption[false]{bsmainpagenumberline}
\DeclareBoolOption[false]{bsfrontpagenumberline}
\DeclareBoolOption[true]{bsheadrule}
\DeclareBoolOption[true]{newtxmath}
\DeclareBoolOption[false]{splitbibitem}
\DeclareBoolOption[false]{engtoc}
\DeclareStringOption{fontset}
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{ctexbook}}
\ProcessKeyvalOptions*
\PassOptionsToPackage{no-math}{fontspec}
\ifheu@bachelor
\PassOptionsToClass{twoside}{book}
\fi
\ifheu@master
\PassOptionsToClass{twoside}{book}
\fi
\ifheu@doctor
\PassOptionsToClass{twoside}{book}
\fi
\ifthenelse%
{\equal{\heu@fontset}{}}%
{%
  \PassOptionsToPackage{AutoFakeBold=2}{xeCJK}
}%
{%
  \ifthenelse%
  {\equal{\heu@fontset}{siyuan}}%
  {\relax}%
  {%
    \PassOptionsToPackage{AutoFakeBold=2}{xeCJK}
  }%
  \PassOptionsToClass{fontset=\heu@fontset}{ctexbook}
}%
\LoadClass[a4paper,openany,UTF8,zihao=-4,scheme=plain]{ctexbook}
\ifheu@bachelor\relax\else
  \ifheu@master\relax\else
    \ifheu@doctor\relax\else
      \ifheu@postdoc\relax\else
         \ClassError{heuthesis}%
            {Please specify thesis type in option: \MessageBreak
             type=[bachelor | master | doctor | postdoc]}{}
      \fi
    \fi
  \fi
\fi
\ifheu@zijvregu\relax\else
\ifheu@zijvword\relax\else
\heu@zijvregutrue
\fi
\fi
\ifheu@geometrynewone\relax\else
\ifheu@geometrynewtwo\relax\else
\ifheu@geometrynewno\relax\else
\heu@geometrynewtwotrue
\fi
\fi
\fi
\ifheu@profdegree{
\ifheu@productdevelopment\relax\else
  \ifheu@projectdplanning\relax\else
    \ifheu@projectdesign\relax\else
      \ifheu@applicationresearch\relax\else
        \ClassError{heuthesisartplus}%
        { Please specify researchtype in option:\MessageBreak
          researchtype=[productdevelopment | projectdplanning | projectdesign | applicationresearch]
        }{}
      \fi
    \fi
  \fi
\fi}
\fi
\RequirePackage{etoolbox}
\RequirePackage{ifxetex}
\ifxetex
\else
  \ClassError{heuthesis}%
    {Please use: \MessageBreak
      xelatex}{}
\fi
\RequirePackage{xparse}
\RequirePackage{amsmath}
\RequirePackage[amsmath,thmmarks,hyperref]{ntheorem}
\RequirePackage{amssymb}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Courier New}
\ifheu@newtxmath
  \let\oldencodingdefault\encodingdefault
  \let\oldrmdefault\rmdefault
  \let\oldsfdefault\sfdefault
  \let\oldttdefault\ttdefault
  \def\encodingdefault{T1}
  \renewcommand{\rmdefault}{ntxtlf}
  \renewcommand{\sfdefault}{qhv}
  \renewcommand{\ttdefault}{ntxtt}
  \RequirePackage{newtxmath}
  \let\encodingdefault\oldencodingdefault
  \let\rmdefault\oldrmdefault
  \let\sfdefault\oldsfdefault
  \let\ttdefault\oldttdefault
\fi
\RequirePackage{graphicx}
\RequirePackage{pdfpages}
\includepdfset{fitpaper=true}
\RequirePackage{enumitem}       %使用enumitem宏包,改变列表项的格式
\RequirePackage{environ}
\ifheu@raggedbottom
  \RequirePackage[bottom,perpage,hang]{footmisc}
  \raggedbottom
\else
  \RequirePackage[perpage,hang]{footmisc}
\fi
\RequirePackage{bbding}
\ifheu@pifootnote
  \RequirePackage{pifont}
\fi
\RequirePackage{xeCJKfntef}
\RequirePackage{diagbox}
\RequirePackage{longtable}
\RequirePackage{booktabs}
\heavyrulewidth=1.5pt
\lightrulewidth=1pt
\cmidrulewidth=0.5pt
\RequirePackage{hyperref}
\hypersetup{%
  CJKbookmarks=true,
  linktoc=all,
  bookmarksnumbered=true,
  bookmarksopen=true,
  bookmarksopenlevel=1,
  breaklinks=true,
  colorlinks=false,
  plainpages=false,
  pdfborder=0 0 0}
\urlstyle{same}
\RequirePackage[sort&compress]{gbt7714}
\ifheu@citetwocomma
  \patchcmd{\NAT@citexnum}{\def@NAT@last@yr{-\NAT@penalty}}
  {%
    \ifx\NAT@last@yr\relax
    \def@NAT@last@yr{\@citea}%
    \else
    \def@NAT@last@yr{-\NAT@penalty}%
    \fi
  }%
  {\typeout{Use comma between neighboring citations}}%
  {\ClassError{heuhesisbook}{fail to patch \NAT@citexnum}{}}%
\fi
\ifheu@debug\RequirePackage[showframe]{geometry}\else\RequirePackage{geometry}\fi
\geometry{%版芯定义
  a4paper, % 210 * 297mm
  hcentering,
  ignoreall,
  nomarginpar,
  centering,
  text={160true mm,241true mm},
  left=25true mm,
  head=20true mm,
  headsep=2true mm,
  footskip=0true mm,
  foot=6true mm
}
\ifheu@debug%
\RequirePackage{layout}
\RequirePackage{layouts}
\RequirePackage{lineno}
\fi
\RequirePackage{fancyhdr}
\RequirePackage{tabularx}
\RequirePackage{varwidth}
\RequirePackage{changepage}
\RequirePackage{multicol}
\RequirePackage[below]{placeins}% 允许上一个section的浮动图形出现在下一个section的开始部分,还提供\FloatBarrier命令,使所有未处理的浮动图形立即被处理
\RequirePackage{flafter}        % 使得所有浮动体不能被放置在其浮动环境之前，以免浮动体在引述它的文本之前出现.
\RequirePackage{multirow}       % 使用Multirow宏包，使得表格可以合并多个row格
\ifheu@subcapcenterlast
\PassOptionsToPackage{centerlast}{subfigure}
\fi
\RequirePackage{subfigure}%支持子图 %centerlast 设置最后一行是否居中
\RequirePackage[subfigure]{ccaption} %支持双语标题
\RequirePackage[makeindex]{splitidx}
\newindex[]{china}
\newindex[]{english}
\RequirePackage{xltxtra}
\ifheu@english
\RequirePackage[xindy,order=letter]{glossaries-extra}
\newglossarystyle{engstyle}{%
  \setglossarystyle{long}%
  \renewenvironment{theglossary}%
  {\vspace{-\intextsep}\centering\begin{longtable}{@{\hskip 1cm}p{3.5cm}p{\glsdescwidth}}}%
     {\end{longtable}}}
\setabbreviationstyle[acronym]{long-short}
\else
\RequirePackage{glossaries}
\setacronymstyle{short-long}
\fi

\renewcommand*{\genacrfullformat}[2]{%
  \glsentrylong{#1}%
}
\ifheu@english
\makenoidxglossaries
\else
\makeglossaries
\fi
\ifheu@library\heu@openrightfalse\else\relax\fi
\ifheu@pageempty
  \let\heu@cleardoublepage\cleardoublepage
  \newcommand{\heu@clearemptydoublepage}{%
    \clearpage{\pagestyle{heu@empty}\heu@cleardoublepage}
  }
  \let\cleardoublepage\heu@clearemptydoublepage
\fi
\renewcommand\frontmatter{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  \@mainmatterfalse
  \pagenumbering{Roman}
  \pagestyle{heu@empty}
\ifheu@openright\cleardoublepage\else\clearpage\fi
\ifheu@zijvword
\ziju{0.05416667}
\else
\ifheu@zijvregu
\ziju{0.05580808}
\fi
\fi
}
\renewcommand\mainmatter{%
  \ifheu@doctor%
    \ifheu@library\clearpage\else\cleardoublepage\fi
    \else%
    \clearpage
  \fi%
  \@mainmattertrue
  \pagenumbering{arabic}
  \pagestyle{heu@headings}
}
\renewcommand\backmatter{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  \@mainmattertrue
  \fancyhf{}
  \pagestyle{heu@titleheadings}
 }
\renewcommand\normalsize{%
  \@setfontsize\normalsize{12bp}{\ifheu@glue 20.50394bp \@plus 2.834646bp \@minus 0bp\else 20.50394bp\fi}%
  \abovedisplayskip=8pt
  \abovedisplayshortskip=8pt
  \belowdisplayskip=\abovedisplayskip
  \belowdisplayshortskip=\abovedisplayshortskip}
\def\heu@def@fontsize#1#2{%
  \expandafter\newcommand\csname #1\endcsname[1][1.3]{%
    \fontsize{#2}{##1\dimexpr #2}\selectfont}}
\heu@def@fontsize{dachu}{58.0bp}
\heu@def@fontsize{chuhao}{42.2bp}
\heu@def@fontsize{xiaochu}{36.1bp}
\heu@def@fontsize{yihao}{26.1bp}
\heu@def@fontsize{xiaoyi}{24.1bp}
\heu@def@fontsize{erhao}{22.1bp}
\heu@def@fontsize{xiaoer}{18.1bp}
\heu@def@fontsize{sanhao}{16.1bp}
\heu@def@fontsize{xiaosan}{15.1bp}
\heu@def@fontsize{sihao}{14.1bp}
\heu@def@fontsize{banxiaosi}{13.0bp}
\heu@def@fontsize{xiaosi}{12.1bp}
\heu@def@fontsize{dawu}{11.0bp}
\heu@def@fontsize{wuhao}{10.5bp}
\heu@def@fontsize{xiaowu}{9.0bp}
\heu@def@fontsize{liuhao}{7.5bp}
\heu@def@fontsize{xiaoliu}{6.5bp}
\heu@def@fontsize{qihao}{5.5bp}
\heu@def@fontsize{bahao}{5.0bp}
\let\heu@headrule\headrule
\fancypagestyle{heu@empty}{%
  \fancyhf{}
  \let\headrule\heu@headrule%
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{heu@headings}{%
  \fancyhf{}
  \ifheu@english
    \fancyhead[C]{\wuhao[0]\leftmark}
    \fancyfoot[C]{\wuhao\thepage}
    \renewcommand{\headrule}{
      \vskip 0.6mm
      \hrule\@height0.5mm\@width\headwidth
      \vskip 0.3mm
      \hrule\@height0.8mm\@width\headwidth
    }
  \else
    \ifheu@postdoc
      \fancyhead[CO]{\songti\wuhao[0]\leftmark}
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@postdoc@documenttitlenospace}
    \fi
    \ifheu@doctor
      \fancyhead[CO]{\songti\wuhao[0]\leftmark}
      \ifheu@harbin
        \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@cdegree\heu@cthesisname}
      \fi
      \ifheu@qingdao
        \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@qingdaocampus\heu@cdegree\heu@cthesisname}
      \fi
      \ifheu@yantai
        \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@yantaicampus\heu@cdegree\heu@cthesisname}
      \fi
    \fi
    \ifheu@master
      \fancyhead[CO]{\songti\wuhao[0]\leftmark}
      \ifheu@harbin
        \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@cdegree\heu@cthesisname}
      \fi
      \ifheu@qingdao
        \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@qingdaocampus\heu@cthesisname}
      \fi
      \ifheu@yantai
        \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@yantaicampus\heu@cdegree\heu@cthesisname}
      \fi
    \fi
    \ifheu@bachelor
      \fancyhead[CO]{\songti\wuhao[0]\leftmark}
      \ifheu@harbin
        \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@bachelor@cxuewei\heu@bachelor@cthesisname}
      \fi
      \ifheu@qingdao
        \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@qingdaocampus\heu@bachelor@cxuewei\heu@bachelor@cthesisname}%
      \fi
      \ifheu@yantai
        \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@yantaicampus\heu@bachelor@cxuewei\heu@bachelor@cthesisname}
      \fi
    \fi
    \fancyfoot[C]{\wuhao\thepage}
    \renewcommand{\headrule}{
    \vskip 0.6mm
    \hrule\@height0.5mm\@width\headwidth
    \vskip 0.3mm
    \hrule\@height0.8mm\@width\headwidth
    }
  \fi
  \renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{heu@titleheadings}%
{%
  \fancyhead{} %
  \fancyhead[CO]{\songti\wuhao[0]\heu@ctitle}
  \ifheu@postdoc
    \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@postdoc@documenttitlenospace}
  \fi
  \ifheu@doctor
    \ifheu@harbin
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@cdegree\heu@cthesisname}
    \fi
    \ifheu@qingdao
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@qingdaocampus\heu@cdegree\heu@cthesisname}
    \fi
    \ifheu@yantai
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@yantaicampus\heu@cdegree\heu@cthesisname}
    \fi
  \fi
  \ifheu@master
    \ifheu@harbin
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@cdegree\heu@cthesisname}
    \fi
    \ifheu@qingdao
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@qingdaocampus\heu@cthesisname}
    \fi
    \ifheu@yantai
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@yantaicampus\heu@cdegree\heu@cthesisname}
    \fi
  \fi
  \ifheu@bachelor
    \ifheu@harbin
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@bachelor@cxuewei\heu@bachelor@cthesisname}
    \fi
    \ifheu@qingdao
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@qingdaocampus\heu@bachelor@cxuewei\heu@bachelor@cthesisname}%
    \fi
    \ifheu@yantai
      \fancyhead[CE]{\songti\wuhao[0]\heu@cschoolname\heu@yantaicampus\heu@bachelor@cxuewei\heu@bachelor@cthesisname}
    \fi
  \fi
  \fancyfoot[C]{\wuhao\thepage}       % 只显示号码
  \renewcommand{\headrulewidth}{0pt}  % 页眉线粗细
  \renewcommand{\footrulewidth}{0pt}  % 页脚线粗细
  \renewcommand{\headrule}{
    \vskip 0.6mm
    \hrule\@height0.5mm\@width\headwidth
    \vskip 0.3mm
    \hrule\@height0.8mm\@width\headwidth
  }
}
\AtBeginDocument{%
  \pagestyle{heu@empty}
  \renewcommand{\chaptermark}[1]{\@mkboth{\CTEXthechapter\enspace#1}{}}}
\ctexset{%
  punct=quanjiao,
  space=auto,
  autoindent=true}
\setlist{nosep}
\def\heu@textcircled#1{%
  \ifnum\value{#1} >9
    \ClassError{heuthesis}%
      {Too many footnotes in this page.}{Keep footnote less than 10.}
  \fi
  \ifheu@pifootnote%
    \ding{\the\numexpr\value{#1}+171\relax}%
  \else%
    \textcircled{\xiaoliu\arabic{#1}}%
  \fi}
\renewcommand{\thefootnote}{\heu@textcircled{footnote}}
\renewcommand{\thempfootnote}{\heu@textcircled{mpfootnote}}
\def\footnoterule{\vskip-3\p@\hrule\@width0.3\textwidth\@height0.4\p@\vskip2.6\p@}
\let\heu@footnotesize\footnotesize
\renewcommand\footnotesize{\heu@footnotesize\xiaowu[1.5]}
\footnotemargin1.5em\relax
\let\heu@makefnmark\@makefnmark
\def\heu@@makefnmark{\hbox{{\normalfont\@thefnmark}}}
\pretocmd{\@makefntext}{\let\@makefnmark\heu@@makefnmark}{}{}
\apptocmd{\@makefntext}{\let\@makefnmark\heu@makefnmark}{}{}
\allowdisplaybreaks[4]
\predisplaypenalty=0  %公式之前可以换页，公式出现在页面顶部
\postdisplaypenalty=0
\renewcommand\theequation{\ifnum \c@chapter>\z@ \thechapter-\fi\@arabic\c@equation}
\def\make@df@tag{\@ifstar\heu@make@df@tag@@\make@df@tag@@@}
\def\heu@make@df@tag@@#1{\gdef\df@tag{\heu@maketag{#1}\def\@currentlabel{#1}}}
\iffalse
\ifheu@bachelor
  \def\heu@maketag#1{\maketag@@@{%
    (\ignorespaces\text{\equationname\hskip0.5em}#1\unskip\@@italiccorr)}}
  \def\tagform@#1{\maketag@@@{%
    (\ignorespaces\text{\equationname\hskip0.5em}#1\unskip\@@italiccorr)\equcaption{#1}}}
\fi
\fi
\def\heu@maketag#1{\maketag@@@{(\ignorespaces #1\unskip\@@italiccorr)}}
\def\tagform@#1{\maketag@@@{(\ignorespaces #1\unskip\@@italiccorr)\equcaption{#1}}}
\renewcommand{\eqref}[1]{\textup{(\ref{#1})}}
\ifheu@postdoc
\setlength{\intextsep}{10.5bp}
\setlength{\textfloatsep}{10.5bp}
\setlength{\floatsep}{10.5bp}
\else
\setlength{\intextsep}{\ifheu@glue 8.50398bp \@plus 2.83465bp \@minus 0bp\else 8.50398bp\fi}
\setlength{\textfloatsep}{\ifheu@glue 8.50398bp \@plus 2.83465bp \@minus 0bp\else 8.50398bp\fi}
\setlength{\floatsep}{\ifheu@glue 12bp \@plus 2.83465bp \@minus 0bp\else 12bp\fi}
\fi
\g@addto@macro\appendix{\renewcommand*{\thefigure}{\thechapter.\arabic{figure}}}
\g@addto@macro\appendix{\renewcommand*{\thetable}{\thechapter.\arabic{table}}}
\g@addto@macro\appendix{\renewcommand*{\theequation}{\thechapter.\arabic{equation}}}
\renewcommand{\textfraction}{0.15}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.65}
\renewcommand{\floatpagefraction}{0.60}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \wuhao\sbox\@tempboxa{#1\ifheu@bachelor\hskip\ccwd\else\enskip\fi#2}%
  \ifdim \wd\@tempboxa >\hsize
    \ifheu@capcenterlast%
      \vskip 6.3bp%
        {\setbox0=\vbox{#1\ifheu@bachelor\hskip\ccwd\else\enskip\fi#2}
           \setbox1=\vbox{%
             \unvbox0
             \setbox2=\lastbox
             \hbox to \textwidth{\hfill\unhcopy2 \unskip\unskip\hfill}
            }
          \unvbox1}
      \else%
        #1\ifheu@bachelor\hskip\ccwd\else\enskip\fi#2%
      \fi%
    \par
  \else
  \global \@minipagefalse
  \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
\vskip\belowcaptionskip}
\long\def\LT@makecaption#1#2#3{%
  \LT@mcol\LT@cols c{\hbox to\z@{\hss\parbox[t]\LTcapwidth{%
    \wuhao\sbox\@tempboxa{#1{#2\ifheu@bachelor\hskip\ccwd\else\enskip\fi }#3}%
    \ifdim\wd\@tempboxa>\hsize
      \ifheu@capcenterlast%
        \vskip 6.3bp%
        {\setbox0=\vbox{#1{#2\ifheu@bachelor\hskip\ccwd\else\enskip\fi }#3}
          \setbox1=\vbox{%
             \unvbox0
             \setbox2=\lastbox
             \hbox to \textwidth{\hfill\unhcopy2 \unskip\unskip\hfill}
          }
        \unvbox1}
      \else%
        #1{#2\ifheu@bachelor\hskip\ccwd\else\enskip\fi }#3%
      \fi%
      \par
    \else
      \global \@minipagefalse
      \hbox to\hsize{\hfil\box\@tempboxa\hfil}
    \fi
    \endgraf\vskip\belowcaptionskip}%
  \hss}}}
\renewcommand*{\longbionenumcaption}[7]{%
\@if@contemptyarg{#1}{\caption{#2}}{\caption[#1]{#2}}%
\global\let\@cont@oldtablename\tablename
\gdef\tablename{#3}
\global\let\LT@c@ption\@cont@LT@nonumintoc
\\[#6]
\@if@contemptyarg{#4}{\caption{#5}}{\caption[#4]{#5}}%
\global\let\tablename\@cont@oldtablename
\global\let\LT@c@ption\@cont@oldLT@c@ption
\vspace{#7}}
\let\heu@LT@array\LT@array
\def\LT@array{\wuhao\heu@LT@array} % set default font size
\newcommand{\ltfontsize}[1]{\def\LT@array{#1\heu@LT@array}}
\renewcommand{\thesubtable}{(\alph{subtable})}
\renewcommand{\thefigure}{\arabic{chapter}.\arabic{figure}}%使图编号为 7.1 的格式 %\protect{~}
\renewcommand{\thesubfigure}{(\alph{subfigure})}%使子图编号为 (a) 的格式
\renewcommand{\p@subfigure}{\thefigure~} %使子图引用为 7.1(a) 的格式，母图编号和子图编号之间用~加一个空格
\renewcommand{\thetable}{\arabic{chapter}.\arabic{table}}%使表编号为 7.1 的格式
\setitemize{leftmargin=0em,itemsep=0em,partopsep=0em,parsep=0em,topsep=0em,itemindent=3em}
\setenumerate{leftmargin=0em,itemsep=0em,partopsep=0em,parsep=0em,topsep=0em,itemindent=3.5em}
\newcommand{\citeup}[1]{\textsuperscript{\cite{#1}}}
\captionnamefont{\wuhao}
\captiontitlefont{\wuhao}
\renewcommand{\subcapsize}{\wuhao}
\setlength{\abovecaptionskip}{0pt}%为了双标题之间的间距，不能设置
\setlength{\belowcaptionskip}{0pt}
\newcounter{pubctr} %自定义新计数器
\newenvironment{publist}{%%定义新环境
\begin{list}{[\arabic{pubctr}]} %%标签格式
    {
     \usecounter{pubctr}
     \setlength{\leftmargin}{1.7em}   % 左边界 \leftmargin =\itemindent + \labelwidth + \labelsep
     \setlength{\itemindent}{0em}     % 标号缩进量
     \setlength{\labelsep}{0.5em}     % 标号和列表项之间的距离,默认0.5em
     \setlength{\rightmargin}{0em}    % 右边界
     \setlength{\topsep}{0ex}         % 列表到上下文的垂直距离
     \setlength{\parsep}{0ex}         % 段落间距
     \setlength{\itemsep}{0ex}        % 标签间距
     \setlength{\listparindent}{0pt} % 段落缩进量
    }}
{\end{list}}
\renewtheoremstyle{plain}
{\item[\hskip\labelsep \theorem@headerfont ##1\ ##2\theorem@separator]}
{\item[\hskip\labelsep \theorem@headerfont ##1\ ##2\ ##3\theorem@separator]}
\theorembodyfont{\songti\rmfamily}
\theoremheaderfont{\heiti\rmfamily}
\theoremsymbol{$\square$}
\setlength{\theorempreskipamount}{0pt}
\setlength{\theorempostskipamount}{-2pt}
\setlength{\parindent}{2em}
\arraycolsep=1.6pt
\def\heu@title@font{%
  \ifheu@arialtitle\sffamily\else\heiti\fi}
\newcommand\heu@chapter@titleformat[1]{%开启悬挂缩进选项
  \ifheu@english
  \MakeUppercase{#1}
  \else
    \ifthenelse%
    {\equal{#1}{\eabstractcname}}%
    {\bfseries #1}%
    % 实现章标题的居中加悬挂缩进，注意，此处一定是\CTEX@chaptername\CTEX@chapter@aftername, 否则是英文标题长度
    {\ifheu@chapterhang\settowidth{\hangindent}{\CTEX@chaptername\CTEX@chapter@aftername}\hangafter=1\fi#1}
  \fi
}
\newcommand\heu@chapter@engformat[1]{%开启悬挂缩进选项
  \begin{center}
    \heu@title@font
    \if@mainmatter
    \xiaoer[1.57481]
    \else
    \sanhao[1.57481]
    \fi\bfseries
    #1
  \end{center}}
\newcommand\heu@chapter@engnameformat[1]{%开启悬挂缩进选项
  \begin{center}
    \MakeUppercase{#1}
  \end{center}}
\newcommand\heu@chapter@postdocformat[1]{\centering\heu@title@font\xiaoer[1.57481]\textbf{#1}}
\renewcommand\@afterheading{%
  \@nobreaktrue
  \everypar{%
    \if@nobreak
      \@nobreakfalse
      \clubpenalty 1
      \if@afterindent \else
        {\setbox\z@\lastbox}%
      \fi
    \else
      \clubpenalty 1
      \everypar{}%
    \fi}}
\ifheu@english
  \ctexset{%
    chapter={
      afterindent=true,
      pagestyle={heu@headings},
      beforeskip={28.34646bp},%一个空行 1.57481 × 18
      afterskip={28.74646bp},%0.8应该不计算间距 0.8 × 18 + 0.57481×18
      aftername={\par},
      format={\heu@chapter@engformat},
      nameformat={\heu@chapter@engnameformat},%\center 会影响之后全局
      numberformat=\relax,
      titleformat={\heu@chapter@titleformat},
      fixskip=true, % 添加这一行去除默认间距
      %hang=true,
    },
    section={
      afterindent=true,
      beforeskip={\ifheu@glue 19.84252bp \@plus 2.834646bp \@minus 0bp \else 19.84252bp \fi},%上下空0.5行
      afterskip={\ifheu@glue 19.84252bp \@plus 2.834646bp \@minus 0bp \else 19.84252bp \fi},
      format={\heu@title@font\ifheu@glue\fontsize{15bp}{21bp \@plus 1.677267bp \@minus 1.157391bp}\else\fontsize{15bp}{21bp}\fi\selectfont\bfseries},
      aftername=\enspace,
      fixskip=true,
      break={},
    },
    subsection={
      afterindent=true,
      beforeskip={\ifheu@glue 17.00787bp \@plus 2.834646bp \@minus 0bp \else 17.00787bp \fi},
      afterskip={\ifheu@glue 17.00787bp \@plus 2.834646bp \@minus 0bp \else 17.00787bp \fi},
      format={\heu@title@font\ifheu@glue\fontsize{14bp}{18bp \@plus 1.842609bp \@minus 0.9920497bp}\else\fontsize{14bp}{18bp}\fi\selectfont\bfseries},
      aftername=\enspace,
      fixskip=true,
      break={},
    },
    subsubsection={
      afterindent=true,
      beforeskip={\ifheu@glue 8.503937bp \@plus 2.834646bp \@minus 0bp \else 8.503937bp \fi},
      afterskip={\ifheu@glue 8.503937bp \@plus 2.834646bp \@minus 0bp \else 8.503937bp \fi},
      format={\heu@title@font\normalsize},
      aftername=\enspace,
      fixskip=true,
      break={},
    },
    paragraph/afterindent=true,
    subparagraph/afterindent=true
  }
\else
\ifheu@postdoc
  \ctexset{%
    chapter={
      afterindent=true,
      pagestyle={heu@headings},
      beforeskip={28.34646bp},%一个空行 1.57481 × 18
      afterskip={28.74646bp},%0.8应该不计算间距 0.8 × 18 + 0.57481×18
      aftername=\enspace,
      format={\heu@chapter@postdocformat},%\center 会影响之后全局
      nameformat=\relax,
      numberformat=\relax,
      titleformat=\heu@chapter@titleformat,
      fixskip=true, % 添加这一行去除默认间距
      %hang=true,
    },
    section={
      afterindent=true,
      beforeskip={\ifheu@glue 19.84252bp \@plus 2.834646bp \@minus 0bp \else 19.84252bp \fi},%上下空0.5行
      afterskip={\ifheu@glue 19.84252bp \@plus 2.834646bp \@minus 0bp \else 19.84252bp \fi},
      format={\heu@title@font\ifheu@glue\fontsize{15bp}{21bp \@plus 1.677267bp \@minus 1.157391bp}\else\fontsize{15bp}{21bp}\fi\selectfont\bfseries},
      aftername=\enspace,
      fixskip=true,
      break={},
    },
    subsection={
      afterindent=true,
      beforeskip={\ifheu@glue 17.00787bp \@plus 2.834646bp \@minus 0bp \else 17.00787bp \fi},
      afterskip={\ifheu@glue 17.00787bp \@plus 2.834646bp \@minus 0bp \else 17.00787bp \fi},
      format={\heu@title@font\ifheu@glue\fontsize{14bp}{18bp \@plus 1.842609bp \@minus 0.9920497bp}\else\fontsize{14bp}{18bp}\fi\selectfont\bfseries},
      aftername=\enspace,
      fixskip=true,
      break={},
    },
    subsubsection={
      afterindent=true,
      beforeskip={\ifheu@glue 8.503937bp \@plus 2.834646bp \@minus 0bp \else 8.503937bp \fi},
      afterskip={\ifheu@glue 8.503937bp \@plus 2.834646bp \@minus 0bp \else 8.503937bp \fi},
      format={\heu@title@font\normalsize},
      aftername=\enspace,
      fixskip=true,
      break={},
    },
    paragraph/afterindent=true,
    subparagraph/afterindent=true
  }
\else
  \ctexset{%
    chapter={
      afterindent=true,
      pagestyle={heu@headings},
      beforeskip={28.34646bp},%一个空行 1.57481 × 18
      afterskip={28.74646bp},%0.8应该不计算间距 0.8 × 18 + 0.57481×18
      aftername=\enspace,
      format={\centering\heu@title@font\xiaoer[1.57481]},%\center 会影响之后全局
      nameformat=\relax,
      numberformat=\relax,
      titleformat=\heu@chapter@titleformat,
      fixskip=true, % 添加这一行去除默认间距
      %hang=true,
    },
    section={
      afterindent=true,
      beforeskip={\ifheu@glue 19.84252bp \@plus 2.834646bp \@minus 0bp \else 19.84252bp \fi},%上下空0.5行
      afterskip={\ifheu@glue 19.84252bp \@plus 2.834646bp \@minus 0bp \else 19.84252bp \fi},
      format={\heu@title@font\ifheu@glue\fontsize{15bp}{21bp \@plus 1.677267bp \@minus 1.157391bp}\else\fontsize{15bp}{21bp}\fi\selectfont},
      aftername=\enspace,
      fixskip=true,
      break={},
    },
    subsection={
      afterindent=true,
      beforeskip={\ifheu@glue 17.00787bp \@plus 2.834646bp \@minus 0bp \else 17.00787bp \fi},
      afterskip={\ifheu@glue 17.00787bp \@plus 2.834646bp \@minus 0bp \else 17.00787bp \fi},
      format={\heu@title@font\ifheu@glue\fontsize{14bp}{18bp \@plus 1.842609bp \@minus 0.9920497bp}\else\fontsize{14bp}{18bp}\fi\selectfont},
      aftername=\enspace,
      fixskip=true,
      break={},
    },
    subsubsection={
      afterindent=true,
      beforeskip={\ifheu@glue 8.503937bp \@plus 2.834646bp \@minus 0bp \else 8.503937bp \fi},
      afterskip={\ifheu@glue 8.503937bp \@plus 2.834646bp \@minus 0bp \else 8.503937bp \fi},
      format={\heu@title@font\normalsize},
      aftername=\enspace,
      fixskip=true,
      break={},
    },
    paragraph/afterindent=true,
    subparagraph/afterindent=true
  }
\fi
\fi
\ifheu@english
\NewDocumentCommand{\heu@appendix@chapter}{s o m o}{%
  \IfBooleanT{#1}%
  {
    \phantomsection
    \IfValueTF{#2}{
      \markboth{#2}{#2}
    }{
      \markboth{#3}{#3}
    }

    \ifthenelse%
    {\equal{#3}{\cabstractcname}}%
    {\relax}
    {\addcontentsline{toc}{chapter}{\texorpdfstring{\ifheu@arialtitle\sffamily\heiti\else\heiti\fi #3}{#3}}}

    \IfValueT{#4}{\addcontentsline{toe}{chapter}{\texorpdfstring{\bfseries #4}{#4}}}
    \heu@chapter*{#3}
  }
}
\else
\NewDocumentCommand{\heu@appendix@chapter}{s o m o}{%
  \IfBooleanT{#1}%
  {
    \phantomsection
    \IfValueTF{#2}{
      \markboth{#2}{#2}
    }{
      \markboth{#3}{#3}
    }
    \ifthenelse%
      {\equal{#3}{\eabstractcname}}%
      {\addcontentsline{toc}{chapter}{\texorpdfstring{\ifheu@arialtitle\sffamily\heiti\else\heiti\fi \ifheu@absupper\MakeUppercase{#3}\else#3\fi}{#3}}}
      {\addcontentsline{toc}{chapter}{\texorpdfstring{\ifheu@arialtitle\sffamily\heiti\else\heiti\fi #3}{#3}}}
    \IfValueT{#4}{\addcontentsline{toe}{chapter}{\texorpdfstring{\bfseries #4}{#4}}}
    \heu@chapter*{#3}
  }
}
\fi
\newcommand{\BiAppChapter}[2]    % 该附录命令适用于有章节的完整附录
{\phantomsection
 \chapter{#1}
 \addcontentsline{toe}{chapter}{\texorpdfstring{\bfseries \xiaosi Appendix \thechapter~~#2}{Appendix \thechapter~~#2}}
}
\let\heu@chapter\chapter
\RenewDocumentCommand{\chapter}{s o m o}{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi\phantomsection%
  \IfBooleanTF{#1}%
  {%    if \chapter*
    \IfNoValueTF{#2}%
    {\heu@chapter*{#3}}%
    {\heu@chapter*[#2]{#3}}%
    \IfValueT{#4}{%
            \addcontentsline{toe}{chapter}{\texorpdfstring{\bfseries #4}{#4}}
    }
  }%
  {%    if \chapter
    \IfNoValueTF{#2}%
    {\heu@chapter{#3}}%
    {\heu@chapter[#2]{#3}}%
    \IfValueT{#4}{%
            \addcontentsline{toe}{chapter}{\texorpdfstring{\bfseries\relax Chapter \thechapter\hspace{0.5em}\ignorespaces #4}{Chapter \thechapter\hspace{0.5em}\ignorespaces #4}}
    }
  }
}
\let\heu@section\section
\RenewDocumentCommand\section{s o m o}{
  \IfBooleanTF{#1}%
  {%    if \section*
    \IfNoValueTF{#2}%
    {\heu@section*{#3}}%
    {\heu@section*[#2]{#3}}%
    \IfValueT{#4}{%
      \addcontentsline{toe}{section}{#4}
    }
  }%
  {%    if \section
    \IfNoValueTF{#2}%
    {\heu@section{#3}}%
    {\heu@section[#2]{#3}}%
    \IfValueT{#4}{%
    \addcontentsline{toe}{section}{\protect\numberline{\csname thesection\endcsname}\ignorespaces #4}
    }
  }
}
\let\heu@subsection\subsection
\RenewDocumentCommand\subsection{s o m o}{
  \IfBooleanTF{#1}%
  {%    if \subsection*
    \IfNoValueTF{#2}%
    {\heu@subsection*{#3}}%
    {\heu@subsection*[#2]{#3}}%
    \IfValueT{#4}{%
      \addcontentsline{toe}{subsection}{#4}
    }
  }%
  {%    if \subsection
    \IfNoValueTF{#2}%
    {\heu@subsection{#3}}%
    {\heu@subsection[#2]{#3}}%
    \IfValueT{#4}{%
    \addcontentsline{toe}{subsection}{\protect\numberline{\csname thesubsection\endcsname}\ignorespaces #4}
    }
  }
}
\let\heu@subsubsection\subsubsection
\RenewDocumentCommand\subsubsection{s o m o}{
  \IfBooleanTF{#1}%
  {%    if \subsubsection*
    \IfNoValueTF{#2}%
    {\heu@subsubsection*{#3}}%
    {\heu@subsubsection*[#2]{#3}}%
    \IfValueT{#4}{%
      \addcontentsline{toe}{subsubsection}{#4}
    }
  }%
  {%    if \subsubsection
    \IfNoValueTF{#2}%
    {\heu@subsubsection{#3}}%
    {\heu@subsubsection[#2]{#3}}%
    \IfValueT{#4}{%
    \addcontentsline{toe}{subsubsection}{\protect\numberline{\csname thesubsubsection\endcsname}\ignorespaces #4}
    }
  }
}
\def\heu@def@term#1{%
  \define@key{heu}{#1}{\csname #1\endcsname{##1}}
  \expandafter\gdef\csname #1\endcsname##1{%
    \expandafter\gdef\csname heu@#1\endcsname{##1}}
  \csname #1\endcsname{}}

\heu@def@term{statesecrets} %密级
\heu@def@term{cnumber} %编号
\heu@def@term{natclassifiedindex}  %国内图书分类号
\heu@def@term{intclassifiedindex}  %国际图书分类号
\heu@def@term{ctitlecover} %中文标题封面
\heu@def@term{ctitle} %中文标题
\heu@def@term{csubtitle} %中文副标题
\heu@def@term{cxueke} %中文学科
\heu@def@term{cauthor} %中文作者
\heu@def@term{csupervisor} %中文导师
\heu@def@term{cprofessional}%职称
\heu@def@term{csupervisorcaffil}%导师单位
\heu@def@term{cassosupervisor} %中文副导师
\heu@def@term{ccosupervisor}%中文联合导师
\heu@def@term{creviewer}%论文主审人
\heu@def@term{caffil}%中文院系
\heu@def@term{csubject}%中文专业
\heu@def@term{csubmitdate}%论文提交日期
\heu@def@term{coralexdate}%论文答辩日期
\heu@def@term{cdate}%论文封面日期
\heu@def@term{cpositionname}%博士后工作站名称
\heu@def@term{cfinishdate}%完成日期
\heu@def@term{cstartdate}%开始日期
\heu@def@term{cenddate}%博士后期满时间

\heu@def@term{cstudentid}%学号
\heu@def@term{cstudenttype}%学生类型
\heu@def@term{ctitleone}%第一行标题
\heu@def@term{ctitletwo}%第二行标题

\heu@def@term{etitle} %英文标题
\heu@def@term{esubtitle} %英文标题
\heu@def@term{exueke} %英文学科
\heu@def@term{eauthor} %英文作者
\heu@def@term{esupervisor} %英文导师
\heu@def@term{eassosupervisor} %英文副导师
\heu@def@term{ecosupervisor} %英文联合导师
\heu@def@term{eaffil}%英文院系
\heu@def@term{esubject}%英文学科
\heu@def@term{esubmitdate}%英文论文提交日期
\heu@def@term{eoralexdate}%英文论文答辩日期
\heu@def@term{edate}%英文日期
\heu@def@term{estudenttype}%英文学生类型

\newcommand{\heu@@cabstract}[1]{\long\gdef\heu@cabstract{#1}}
\newenvironment{cabstract}{\Collect@Body\heu@@cabstract}{}
\newcommand{\heu@@eabstract}[1]{\long\gdef\heu@eabstract{#1}}
\newenvironment{eabstract}{\Collect@Body\heu@@eabstract}{}
\def\heu@parse@keywords#1{
  \define@key{heu}{#1}{\csname #1\endcsname{##1}}
  \expandafter\gdef\csname heu@#1\endcsname{}
  \expandafter\gdef\csname #1\endcsname##1{
    \@for\reserved@a:=##1\do{
      \expandafter\ifx\csname heu@#1\endcsname\@empty\else
        \expandafter\g@addto@macro\csname heu@#1\endcsname{%
          \ignorespaces\csname heu@#1@separator\endcsname}
      \fi
      \expandafter\expandafter\expandafter\g@addto@macro%
        \expandafter\csname heu@#1\expandafter\endcsname\expandafter{\reserved@a}}}}
\heu@parse@keywords{ckeywords}
\heu@parse@keywords{ekeywords}
\def\heusetup{\kvsetkeys{heu}}
\newlength{\heu@title@width}
\newcommand{\heu@put@title}[2][\heu@title@width]{%
  \begin{CJKfilltwosides}[b]{#1}#2\end{CJKfilltwosides}}

\def\heu@first@titlepage{%
  \ifheu@postdoc\heu@first@titlepage@postdoc\else\ifheu@bachelor\heu@first@titlepage@bachelor\else\heu@first@titlepage@other\fi\fi}

\def\heu@second@titlepage{%
  \ifheu@postdoc\heu@second@titlepage@postdoc\else\ifheu@bachelor\heu@second@titlepage@bachelor\else\heu@second@titlepage@other\fi\fi}

\newcommand{\heu@first@titlepage@postdoc}{
  {\songti\sihao\renewcommand{\arraystretch}{2}\
    \begin{tabular}{@{}r@{}l@{\hspace{7\ccwd}}r@{}l@{\hspace{3em}}r}
      \heu@postdoc@classifiedindex & \underline{\makebox[6\ccwd]{\heu@natclassifiedindex}} & \heu@postdoc@secretlevel & \underline{\makebox[6\ccwd]{\heu@statesecrets}} & \\
      \heu@postdoc@UDC             & \underline{\makebox[6\ccwd]{\heu@intclassifiedindex}} & \heu@postdoc@number      & \underline{\makebox[6\ccwd]{\heu@cnumber}} & \\
    \end{tabular}\renewcommand{\arraystretch}{1}}
  \vspace*{0.5cm}
  \parbox[t][5cm][b]{\textwidth}{\begin{center}\heiti\xiaoer[2]\heu@cschoolname\par
      \heu@postdoc@documenttitle\end{center}}
  \parbox[b][4cm][b]{\textwidth}{
    \songti\xiaoer[2]\renewcommand{\arraystretch}{1.5}
    \begin{center}
      \begin{tabular}{c}
        \makebox[9cm]{\heu@ctitleone}\\
        \hline
        \makebox[9cm]{\heu@ctitletwo}\\
        \hline
      \end{tabular}\renewcommand{\arraystretch}{1}
    \end{center}}
    \begin{center}
      \songti\sihao[2]\centering\heu@cauthor
    \end{center}
  \parbox[b][4.5cm][b]{\textwidth}{
    \songti\sihao[2]\renewcommand{\arraystretch}{1.5}
    \begin{center}
      \begin{tabular}{rl}
        \heu@postdoc@finishdate & \underline{\makebox[6cm]{\heu@cfinishdate}} \\
        \heu@postdoc@submitdate & \underline{\makebox[6cm]{\heu@csubmitdate\hfill}} \\
      \end{tabular}\renewcommand{\arraystretch}{1}
    \end{center}
    }
    \vfill
    {\songti\sihao[2]
     \begin{center}
       \heu@cschoolname%
       %\ifheu@harbin
       %\heu@harbincampus
       %\fi
       \ifheu@yantai
       \heu@yantaicampus
       \fi
       \ifheu@qingdao
       \heu@qingdaocampus
       \fi
    \end{center}
    \begin{center}
      \heu@cdate
    \end{center}}
}

\newcommand{\heu@second@titlepage@postdoc}{
  \ifthenelse%
  {\equal{\heu@fontset}{siyuan}}%
  {\xiaosi[1]\vspace*{0.65em}}%
  {\xiaosi[1]\textcolor[rgb]{1,1,1}{\songti{\heu@motto}}}%
  \vspace*{1.2cm}
  \begin{center}
    \parbox[t][8cm][t]{\textwidth}{
      \begin{center}\songti\sanhao[2]\heu@ctitlecover\end{center}
      \vspace*{16bp}
      \begin{center}\sihao[2]\MakeUppercase{\heu@etitle}\end{center}
    }
    \parbox[t][5cm][t]{\textwidth}{
      \begin{center}\songti\sihao[2]
        \begin{tabularx}{0.6\linewidth}{l@{\heu@postdoc@separator}l}
          \makebox[9em][s]{\heu@postdoc@authorname}   & \heu@cauthor \\
          \makebox[9em][s]{\heu@postdoc@supervisor}   & \heu@csupervisor \\
          \makebox[9em][s]{\heu@postdoc@positionname} & \heu@cpositionname \\
          \makebox[9em][s]{\heu@postdoc@major}        & \heu@csubject \\
          \makebox[9em][s]{\heu@postdoc@startdate}    & \heu@cstartdate \\
          \makebox[9em][s]{\heu@postdoc@enddate}      & \heu@cenddate \\
        \end{tabularx}
      \end{center}
    }
  \end{center}
  \vfill
  {\songti\sihao[2]
    \begin{center}
      \heu@cschoolname%
      %\ifheu@harbin
      %\heu@harbincampus
      %\fi
      \ifheu@yantai
      \heu@yantaicampus
      \fi
      \ifheu@qingdao
      \heu@qingdaocampus
      \fi
    \end{center}
    \begin{center}
      \heu@cdate
    \end{center}}}

\newcommand{\heu@first@titlepage@bachelor}{
\ifthenelse%
  {\equal{\heu@fontset}{siyuan}}%
  {\xiaosi[1]\textcolor[rgb]{1,1,1}{\songti{\heu@motto}}}%
  \vspace*{0.1cm}
  \begin{center}
  {\songti\xiaosi
    \begin{tabular}{@{\qquad\qquad}@{}r@{}l@{}l@{\qquad}}
     &  & \\
     &  &
    \end{tabular}}\hfill
  {\songti\xiaosi
    \begin{tabular}{@{}r@{：}l@{}l@{\qquad}}
    \heu@bachelor@cstudentidtitle & \underline{\makebox[2.5cm]{\heu@cstudentid}}   & \\
    \heu@bachelor@secretlevel     & \underline{\makebox[2.5cm]{\heu@statesecrets}} &
    \end{tabular}}
  \parbox[t][2.0cm][t]{\textwidth}{\begin{center} \end{center} }
  \parbox[t][1.2cm][t]{\textwidth}{
    \begin{center}\xiaoer[1]\songti{\heu@cschoolname\heu@bachelor@cxuewei\heu@bachelor@cthesisname}\end{center}}
  \parbox[t][1.2cm][t]{\textwidth}{}
  \parbox[t][7.8cm][t]{\textwidth}{%
    \begin{center}\erhao\heiti\heu@ctitlecover\end{center}
    \ifheu@subtitle\begin{center}\hspace{-4em}\xiaoer\heiti\pozhehao\heu@csubtitle\end{center}\fi}
  \parbox[t][7.4cm][t]{\textwidth}{
  {\xiaosan
    \setlength{\heu@title@width}{7em}
    \begin{center} \renewcommand{\arraystretch}{1.62} \songti
    \begin{tabular}{l@{\heu@title@csep}l}
    {\songti\heu@put@title{\heu@bachelor@caffiltitle}}      & \heu@caffil \\
    {\songti\heu@put@title{\heu@bachelor@cmajortitlecov}}   & \heu@csubject \\
    {\songti\heu@put@title{\heu@bachelor@cstudenttitle}}    & \heu@cauthor\\
    {\songti\heu@put@title{\heu@bachelor@csupervisortitle}} & \heu@csupervisor\quad\heu@cprofessional
    \end{tabular} \renewcommand{\arraystretch}{1}
    \end{center}}}
  \parbox[t][1.4cm][t]{\textwidth}{
  \begin{center}\kaishu\xiaoer{\heu@cschoolname}\end{center}}
  {\songti\xiaosan{\heu@cdate}}
  \end{center}
}

\newcommand{\heu@second@titlepage@bachelor}{
  \vspace*{0.5cm}
  {\songti\xiaosi
    \begin{tabular}{@{\qquad\qquad}@{}r@{}l@{}l@{\qquad}}
    &  & \\
    &  &
    \end{tabular}}\hfill
  {\songti\xiaosi
    \begin{tabular}{@{}r@{：}l@{}l@{\qquad}}
    \heu@bachelor@cstudentidtitle & \underline{\makebox[2.5cm]{\heu@cstudentid}}   & \\
    \heu@bachelor@secretlevel     & \underline{\makebox[2.5cm]{\heu@statesecrets}} &
    \end{tabular}}
  \vspace*{1.5cm}
  \begin{center}
    \parbox[t][8.0cm][t]{\textwidth}
    {\heiti\xiaosan
        \begin{center}\erhao\heiti\heu@ctitlecover\end{center}
      \ifheu@subtitle\begin{center}\hspace{-4em}\xiaoer\heiti\pozhehao\heu@csubtitle\end{center}\fi
      \begin{center}\erhao{\heu@etitle}\end{center}
      \begin{center}\xiaoer\textbf{\ifheu@subtitle\heu@title@esep\heu@esubtitle\fi}\end{center}
    }
  \parbox[t][9.8cm][b]{\textwidth}
  {\heiti\xiaosan
    \setlength{\heu@title@width}{6em}
    \begin{center} \renewcommand{\arraystretch}{1.5} \songti
    \begin{tabular}{l@{\heu@title@csep}l}
    \textbf{\heu@put@title{\heu@bachelor@cstudenttitle}}        & \heu@cauthor\\
    \textbf{\heu@put@title{\heu@bachelor@ccolegetitle}}         & \heu@caffil\\
    \textbf{\heu@put@title{\heu@bachelor@cmajortitlesec}}       & \heu@csubject\\
    \textbf{\heu@put@title{\heu@bachelor@csupervisortitle}}     & \heu@csupervisor\\
    \textbf{\heu@put@title{\heu@bachelor@cprofessionaltitle}}   & \heu@cprofessional\\
    \textbf{\heu@put@title{\heu@bachelor@cupervisoraffiltitle}} & \heu@csupervisorcaffil\\
    \textbf{\heu@put@title{\heu@csubmitdatatitle}}              & \heu@csubmitdate\\
    \textbf{\heu@put@title{\heu@coralexdatetitle}}              & \heu@coralexdate\\
    \textbf{\heu@put@title{\heu@cschoolnametitle}}              & \heu@cschoolname
    \end{tabular} \renewcommand{\arraystretch}{1}
    \end{center}
  }
  \end{center}
}

\newcommand{\heu@first@titlepage@other}{
\ifthenelse%
{\equal{\heu@fontset}{siyuan}}%
{\xiaosi[1]\textcolor[rgb]{1,1,1}{\songti{\heu@motto}}}%
 \vspace*{0.1cm}
\begin{center}
  {\songti\xiaosi
    \begin{tabular}{@{\qquad}@{}r@{：}l@{}l@{\qquad}}
    \heu@natclassifiedindextitle      & \underline{\makebox[2.5cm]{\heu@natclassifiedindex}} & \\
    \heu@internatclassifiedindextitle & \underline{\makebox[2.5cm]{\heu@intclassifiedindex}} &
    \end{tabular}}\hfill
  {\songti\xiaosi
    \begin{tabular}{@{}r@{：}l@{}l@{\qquad}}
    \heu@secretlevel  & \underline{\makebox[2.5cm]{\heu@statesecrets}} & \\
    \heu@cnumbertitle & \underline{\makebox[2.5cm]{\heu@cnumber}}      &
    \end{tabular}}

  \vspace*{2.0cm}
  \parbox[t][1.2cm][t]{\textwidth}{\xiaoer[1]
  \begin{center}\xiaoer[1]\songti{\heu@cdegree\heu@cthesisname}\end{center}}
  \parbox[t][1.2cm][t]{\textwidth}{\quad}
    \parbox[t][6.0cm][t]{\textwidth}{%
  \begin{center}\erhao\heiti\heu@ctitlecover\end{center}
    \ifheu@subtitle\begin{center}\hspace{-4em}\xiaoer\heiti\pozhehao\heu@csubtitle\end{center}\fi}
    \parbox[t][7.4cm][t]{\textwidth}{
    {\xiaosan
    \setlength{\heu@title@width}{7em}
    \begin{center} \renewcommand{\arraystretch}{1.62} \songti
      \begin{tabular}{l@{\heu@title@csep}l}
      {\songti\heu@put@title{\heu@cauthortitle}}           & \heu@cauthor\\
      {\songti\heu@put@title{\heu@csupervisortitle}}       & \heu@csupervisor\\
      \ifx\heu@cassosupervisor\@empty\else%
        {\songti\heu@put@title{\heu@cassosupervisortitle}} & \heu@cassosupervisor\\
      \fi
      \ifheu@profdegree%
        {\songti\heu@put@title{\heu@ccosupervisortitle}}   & \heu@ccosupervisor \\
        {\songti\heu@put@title{\heu@cprodrgreetitle}}      & \heu@cstudenttype \\
      \else%
        {\songti\heu@put@title{\heu@csubjecttitle}}        & \heu@csubject \\
      \fi
      \ifx\heu@creviewer\@empty\else%
        {\songti\heu@put@title{\heu@creviewertitle}}       & \heu@creviewer
      \fi
      \end{tabular} \renewcommand{\arraystretch}{1}
    \end{center}}}
    \parbox[t][3.0cm][b]{\textwidth}{
  \begin{center}\kaishu\xiaoer{\heu@cschoolname}\end{center}}
  {\songti\xiaosan{\heu@cdate}}
  \end{center}
}

\newcommand{\heu@second@titlepage@other}{
  \begin{center}
    {\songti\xiaosi
      \begin{tabular}{@{\qquad}@{}r@{：}l@{}l@{\qquad}}
    \heu@natclassifiedindextitle      & \underline{\makebox[2.5cm]{\heu@natclassifiedindex}} & \\
    \heu@internatclassifiedindextitle & \underline{\makebox[2.5cm]{\heu@intclassifiedindex}} &
    \end{tabular}}\hfill
    {\songti\xiaosi
      \begin{tabular}{@{}r@{：}l@{}l@{\qquad}}
    \heu@secretlevel & \underline{\makebox[2.5cm]{\heu@statesecrets}} &   \\
    \heu@cnumbertitle & \underline{\makebox[2.5cm]{\heu@cnumber}} &
    \end{tabular}}
  \parbox[t][2.0cm][t]{\textwidth}{\begin{center} \end{center}}
    \parbox[t][1.2cm][t]{\textwidth}{\xiaoer[1]
  \begin{center}\songti\heu@cdegree\heu@cthesisname\end{center}}
  \parbox[t][1.2cm][t]{\textwidth}{\quad}
    \parbox[t][5cm][t]{\textwidth}{\erhao
  \begin{center}\heiti\heu@ctitlecover\end{center}
    \ifheu@subtitle\hspace{-4em}\xiaoer\heiti\pozhehao\heu@csubtitle\fi
    \ifheu@profdegree{\begin{center}\sihao\songti\par\heu@searcetype\end{center}}\else{\par\quad}\fi}
    \parbox[t][9.8cm][b]{\textwidth}
    {\sihao
      \setlength{\heu@title@width}{6em}
      \begin{center} \renewcommand{\arraystretch}{1.5} \songti
    \begin{tabular}{l@{\heu@title@csep}l}
      \textbf{\heu@put@title{\heu@cauthortitle}}           & \heu@cauthor\\
      \textbf{\heu@put@title{\heu@csupervisortitle}}       & \heu@csupervisor\\
      \ifx\heu@cassosupervisor\@empty\else%
        \textbf{\heu@put@title{\heu@cassosupervisortitle}} & \heu@cassosupervisor\\
      \fi
      \ifheu@profdegree%
        \textbf{\heu@put@title{\heu@ccosupervisortitle}}   & \heu@ccosupervisor\\
        \textbf{\heu@put@title{\heu@cprosubjecttile}}      & \heu@csubject\\
        \textbf{\heu@put@title{\heu@ccolegetitle}}         & \heu@caffil\\
      \else%
        \textbf{\heu@put@title{\heu@cdegreetitle}}         & \heu@cdegree\\
        \textbf{\heu@put@title{\heu@csubjecttitle}}        & \heu@csubject\\
        \textbf{\heu@put@title{\heu@caffiltitle}}          & \heu@caffil\\
      \fi
      \textbf{\heu@put@title{\heu@csubmitdatatitle}}       & \heu@csubmitdate\\
      \textbf{\heu@put@title{\heu@coralexdatetitle}}       & \heu@coralexdate\\
      \textbf{\heu@put@title{\heu@cschoolnametitle}}       & \heu@cschoolname
    \end{tabular} \renewcommand{\arraystretch}{1}
    \end{center} }
  \end{center}
}
\newcommand{\emultiline}[2][c]{\renewcommand{\arraystretch}{1}\begin{tabular}[#1]{@{}l@{}}#2\end{tabular} \renewcommand{\arraystretch}{1.3}}
\newcommand{\heu@engcover}{
  {\songti\xiaosi
    \begin{tabular}{@{\qquad}l@{}l@{\qquad}}
    Classified Index:\ \heu@natclassifiedindex & \\
    U.D.C:\ \heu@intclassifiedindex &
    \end{tabular}}\hfill
  {\songti\xiaosi
    \begin{tabular}{@{}r@{ }l@{}l@{\qquad}}
     &  &   \\
     &  &
    \end{tabular}}
  \vspace*{1em}
  \begin{center}
  \parbox[t][1.6cm][t]{\textwidth}{\begin{center} \end{center} }
    \parbox[t][1.2cm][t]{\textwidth}{\xiaoer[1]
      \begin{center}
        \ifheu@profdegree A Thesis for the Degree of {\heu@estudenttype}
        \else A Dissertation for the Degree of {\heu@exueweier}
        \fi
      \end{center}}
    \parbox[t][1.2cm][t]{\textwidth}{\xiaoer[1] \qquad}
    \parbox[t][7cm][t]{\textwidth}{%
    \begin{center}\erhao{\heu@etitle}\end{center}
    \begin{center}\xiaoer{\ifheu@subtitle\heu@title@esep\heu@esubtitle\fi}\end{center}}
    {\xiaosi\renewcommand{\arraystretch}{1.3}
      \begin{tabular}{@{}r@{~}l@{}}
        \textbf{\heu@eauthortitle\heu@title@esep}            & \heu@eauthor\\
        \textbf{\heu@esupervisortitle\heu@title@esep}        & \heu@esupervisor\\
        \ifx\heu@eassosupervisor\@empty\else%
          \textbf{\heu@put@title{\heu@eassosupervisortitle}} & \heu@eassosupervisor\\
        \fi
        \ifheu@profdegree%
          \textbf{\heu@eassosupervisortitle\heu@title@esep}  & \heu@eassosupervisor\\
          \textbf{\heu@ecategorytitle\heu@title@esep}        & \heu@esubject\\
          \textbf{\heu@eaffiltitle\heu@title@esep}           & \heu@eaffil\\
        \else%
            \textbf{\heu@edegreetitle\heu@title@esep}        & \heu@edegree\\
          \textbf{\heu@esubjecttitle\heu@title@esep}         & \heu@esubject\\
        \fi
        \textbf{\heu@esubmitdatetitle\heu@title@esep}        & \heu@esubmitdate\\
        \textbf{\heu@eoralexdatetitle\heu@title@esep}        & \heu@eoralexdate\\
        \textbf{\heu@eschoolnametitle\heu@title@esep}        & \heu@eschoolname
      \end{tabular}\renewcommand{\arraystretch}{1}}
  \end{center}
}

\def\makecover{
  \phantomsection
  \pdfbookmark[0]{\heu@ctitle}{ctitle}
  \xiaosi[1]%
  \begin{titlepage}
    \ifheu@yantai
      \ifheu@bachelor
        \relax
      \else
        \heu@first@titlepage
        \ifheu@library\clearpage\else\cleardoublepage\fi
      \fi
    \else
      \heu@first@titlepage
      \ifheu@library\clearpage\else\cleardoublepage\fi
    \fi
    \heu@second@titlepage
    \ifheu@library\clearpage\else\cleardoublepage\fi
    \ifheu@postdoc
    \relax
    \else
    \ifheu@bachelor
    \relax
    \else
    \phantomsection
    \pdfbookmark[0]{\heu@etitle}{etitle}
    \heu@engcover
    \ifheu@library\clearpage\else\cleardoublepage\fi
    \fi
    \fi
  \end{titlepage}
  \normalsize
  \cleardoublepage}
\def\heu@authorization@other{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  %  \heu@appendix@chapter*{\heu@authorization@ctitle}[\heu@authorization@etitle]
  %  \xiaosi[1.5]\vspace{\baselineskip}
  \begin{center}
      \quad\\
      \xiaoer\heiti\heu@cschoolname\\%
      \vspace{0.25\baselineskip}%
      \heu@declarename
  \end{center}
  \vspace{0.25\baselineskip}
  \par\heu@declaretext
  \vspace{\baselineskip}
  \par\hspace{23em}{\wuhao \heu@authorsig}
  \vspace{0.25\baselineskip}
  \par\hspace{23em}{\wuhao \heu@frontdate\heu@datefill}
  \vspace{\baselineskip}
  \begin{center}
      \xiaoer\heiti\heu@cschoolname\\%
      \vspace{0.25\baselineskip}%
      \heu@authorizationtitle
  \end{center}
  \vspace{0.25\baselineskip}
  \par\heu@authorizationtext
  \vspace{\baselineskip}
  \par\hspace{0em}{\wuhao \heu@authorsig} \hspace{17em} {\wuhao \heu@teachersig}
  \vspace{0.5\baselineskip}
  \par\hspace{0em}{\wuhao \heu@frontdate\heu@datefill} \hspace{12.5em}{\wuhao \heu@frontdate\heu@datefill}}
\def\heu@authorization@bachelor{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  \heu@appendix@chapter*{\heu@declarename@bachelor}
  \heu@authorizationtext@bachelor
  \vspace{\baselineskip}
  \par\hspace{6em}\heu@authorsig\hfill\heu@frontdate\heu@datefill}
\NewDocumentCommand{\authorization}{o}{%
  \IfNoValueTF{#1}{%
      \thispagestyle{empty}%
      % 本科不需要原创性和授权页，删掉
      %\ifheu@bachelor\heu@authorization@bachelor\else\heu@authorization@other\fi
      \heu@authorization@other
  }{%
      \ifheu@openright\cleardoublepage\else\clearpage\fi%
      \includepdf[fitpaper=true,pagecommand={%
          \thispagestyle{empty}%
          % 原创性和授权页不加到目录
          %\phantomsection\addcontentsline{toc}{chapter}{\ifheu@bachelor\heu@declarename@bachelor\else\heu@authorization@ctitle\fi}%
          %\ifheu@doctor%
          %  \addcontentsline{toe}{chapter}{\texorpdfstring{\bfseries \heu@authorization@etitle}{\heu@authorization@etitle}}%
          %\fi%
      }]{#1}%
  }%
}
\newbox\heu@kw
\newcommand\heu@put@keywords[2]{%
  \begingroup
    \setbox\heu@kw=\hbox{#1}
    \noindent\hangindent\wd\heu@kw\hangafter1%
    \box\heu@kw#2\par
  \endgroup}
\newcommand{\makeabstract}{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  \ifheu@english
  \addtocontents{toc}{~\hfill\textbf{Page}\par}
  \addtocontents{\ext@table}{{\noindent\sihao\textbf{Table}~\hfill\textbf{Page}}\par}%
  \addtocontents{\ext@figure}{{\noindent\sihao\textbf{Figure}~\hfill\textbf{Page}}\par}%
  \fi
  \pagestyle{heu@titleheadings}
  \pagenumbering{Roman}
  \begin{center}
    ~\\
    \xiaoer\heiti\cabstractcname
  \end{center}\par
  \heu@cabstract
  \vskip12bp
  \heu@put@keywords{\ifheu@postdoc\bfseries\fi\heiti\heu@ckeywords@title}{\heu@ckeywords}
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  \clearpage
  \begin{center}
    ~\\
    \xiaoer\textbf{\eabstractcname}
  \end{center}\par
  \heu@eabstract
  \vskip12bp
  \heu@put@keywords{\textbf{Keywords:\enskip}}{\heu@ekeywords}}
\newenvironment{denotation}[1][2.5cm]{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  \heu@appendix@chapter*{\heu@denotation@ctitle}[\heu@denotation@etitle]
\setcounter{table}{0}
\renewcommand{\thetable}{\arabic{table}}%使表编号为 1 的格式
  }{\renewcommand{\thetable}{\arabic{chapter}.\arabic{table}}%使表编号为 7.1 的格式
\setcounter{table}{0}}%
\def\heu@starttoc#1{% #1: float type, prepend type name in \listof*** entry.
  \let\oldnumberline\numberline
  \def\numberline##1{\oldnumberline{\csname #1name\endcsname\hskip.4em ##1}}
  \@starttoc{\csname ext@#1\endcsname}
  \let\numberline\oldnumberline}
\def\heu@listof#1#2{% #1: float type
  \clearpage
  \begin{center}
    ~\\
    {\xiaoer\heiti{\csname list#1name\endcsname}}
  \end{center}\par
  \heu@starttoc{#1}}

\renewcommand\listoffigures{\heu@listof{figure}{\listfigureename}}
\renewcommand*\l@figure{\addvspace{6bp}\@dottedtocline{1}{0em}{4em}}
\renewcommand\listoftables{\heu@listof{table}{\listtableename}}
\let\l@table\l@figure
\def\ext@equation{loe}
\def\equcaption#1{%
  \addcontentsline{\ext@equation}{equation}%
                  {\protect\numberline{#1}}}
\newcommand\listofequations{\heu@listof{equation}{\listequationename}}
\let\l@equation\l@figure
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\ifheu@bachelor\ifheu@tocfour\setcounter{tocdepth}{3}\fi\fi
\renewcommand\tableofcontents{%
\ifheu@openright\cleardoublepage\else\clearpage\fi
\phantomsection
\markboth{\contentsname}{ccontent}
\begin{center}
  ~\\
  \xiaoer\heiti\contentsname
\end{center}\par
\pdfbookmark[0]{\contentsname}{ccontent}
\normalsize\@starttoc{toc}
\ifheu@doctor
\tableofengcontents
\else
\ifheu@engtoc
\tableofengcontents
\fi
\fi
\ifheu@english
\printnoidxglossary[sort=word,style=engstyle,title={Nomenclature},nopostdot,nonumberlist]
\listoftables
\listoffigures
\fi
}
\ifheu@arialtoc
  \def\heu@toc@font{\sffamily}
\fi
\def\@pnumwidth{4em}%规定中的提前悬挂
\def\@tocrmarg{\@pnumwidth}
\def\@dotsep{1}
\setlength\@tempdima{4em}%
\patchcmd{\@dottedtocline}{#4}{\csname heu@toc@font\endcsname #4}{}{}
\patchcmd{\@dottedtocline}{\hb@xt@\@pnumwidth}{\hbox}{}{}
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    %\vskip 4bp \@plus\p@
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      % numberline is called here, and it uses \@tempdima
      {\ifheu@bachelor\rmfamily\else\csname
        heu@toc@font\endcsname\fi\heiti\ifheu@postdoc\bfseries\fi\ifheu@english\bfseries\fi #1}
      \leaders\hbox{$\m@th\mkern \@dotsep mu\hbox{.}\mkern \@dotsep mu$}\hfill
      \nobreak{\normalfont\normalcolor #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\renewcommand*\l@section{\@dottedtocline{1}{1em}{1.8em}}
\renewcommand*\l@subsection{\@dottedtocline{2}{2em}{2.5em}}
\renewcommand*\l@subsubsection{\@dottedtocline{3}{3\ccwd}{3.1em}}
\def\@dotsep{0.75}           % 定义英文目录的点间距
\setlength\leftmargini {0pt}
\setlength\leftmarginii {0pt}
\setlength\leftmarginiii {0pt}
\setlength\leftmarginiv {0pt}
\setlength\leftmarginv {0pt}
\setlength\leftmarginvi {0pt}

\def\engcontentsname{\bfseries Contents}
\newcommand\tableofengcontents{
  \def\l@chapter{\@dottedtocline{0}{0em}{5em}}%控制英文目录： 细点\@dottedtocline  粗点\@dottedtoclinebold
  \@restonecolfalse
  \chapter*{\engcontentsname  %chapter*上移一行，避免在toc中出现。
  \pdfbookmark[0]{Contents}{econtent}
    \@mkboth{%
  \engcontentsname}{\engcontentsname}}
  \@starttoc{toe}%
  \if@restonecol\twocolumn\fi}
\def\@dotsep{0.75}           % 定义英文目录的点间距
\ctexset{%
  appendix/number=\ifheu@bachelor\arabic{chapter}\else\Alph{chapter}\fi,
}
\let\heu@appendix\appendix
\renewenvironment{appendix}[2]{%
\heu@appendix
\ifheu@bachelor\renewcommand{\thechapter}{\arabic{chapter}}\fi
\ifheu@openright\cleardoublepage\else\clearpage\fi
\renewcommand{\thechapter}{#1} % 修改附录序号
\ifheu@english
\addcontentsline{toc}{chapter}{\heu@appendix@etitle #1 ~ #2}
\begin{center}
  ~\\
  \xiaoer\heiti\heu@appendix@etitle #1 ~ #2
\end{center}\par
\else
\addcontentsline{toc}{chapter}{\heu@appendix@ctitle #1 ~ #2}
\begin{center}
  ~\\
  \xiaoer\heiti\heu@appendix@ctitle #1 ~ #2
\end{center}\par
\fi}{%
  \let\title\@gobble}
\let\title\@gobble

\newlist{translationbib}{enumerate}{1}
\setlist[translationbib]{label=[\arabic*],align=left,nosep,itemsep=6bp,
  leftmargin=10mm,labelsep=!,before=\vspace{0.5\baselineskip}\wuhao[1.3]}

\newenvironment{conclusions}{%
\ifheu@openright\cleardoublepage\else\clearpage\fi
\ifheu@english
\addcontentsline{toc}{chapter}{\heu@conclusion@etitle}
\begin{center}
  ~\\
  \xiaoer\heiti\heu@conclusion@etitle
\end{center}\par
\else
\addcontentsline{toc}{chapter}{\heu@conclusion@ctitle}
\begin{center}
  ~\\
  \xiaoer\heiti\heu@conclusion@ctitle
\end{center}\par
\fi}{}

\newenvironment{acknowledgements}{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  \ifheu@english
  %\heu@appendix@chapter*{\heu@acknowledgement@etitle}
  \addcontentsline{toc}{chapter}{\heu@acknowledgement@etitle}
  \begin{center}
    ~\\
    \xiaoer\heiti\heu@acknowledgement@etitle
  \end{center}\par
  \else
  %\heu@appendix@chapter*{\heu@acknowledgement@ctitle}[\heu@acknowledgement@etitle]
  \addcontentsline{toc}{chapter}{\heu@acknowledgement@ctitle}
  \begin{center}
    ~\\
    \xiaoer\heiti\heu@acknowledgement@ctitle
  \end{center}\par
  \fi}{}

\newenvironment{correspondingaddr}{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  %\heu@appendix@chapter*{\heu@correspondingaddr@ctitle}
  \addcontentsline{toc}{chapter}{\heu@correspondingaddr@ctitle}
  \begin{center}
    ~\\
    \xiaoer\heiti\heu@correspondingaddr@ctitle
  \end{center}\par
  }{}

\newenvironment{resume}{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  %\heu@appendix@chapter*{\heu@resume@ctitle}[\heu@resume@etitle]
  \addcontentsline{toc}{chapter}{\heu@resume@ctitle}
  \begin{center}
    ~\\
    \xiaoer\heiti\heu@resume@ctitle
  \end{center}\par
  }{}

\newenvironment{publication}{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  %\heu@appendix@chapter*{\heu@publication@ctitle}[\heu@publication@etitle]
  \addcontentsline{toc}{chapter}{\heu@publication@ctitle}
  \begin{center}
    ~\\
    \xiaoer\heiti\heu@publication@ctitle
  \end{center}\par
  }{}

\newenvironment{doctorpublication}{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  %\heu@appendix@chapter*{\heu@doctorpublication@ctitle}
  \addcontentsline{toc}{chapter}{\heu@doctorpublication@ctitle}
  \begin{center}
    ~\\
    \xiaoer\heiti\heu@doctorpublication@ctitle
  \end{center}\par
  }{}

\newenvironment{ceindex}{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi
  %\heu@appendix@chapter*{\indexname}[\heu@index@etitle]
  \addcontentsline{toc}{chapter}{\indexname}
  \begin{center}
    ~\\
    \xiaoer\heiti\indexname
  \end{center}\par
  \setlength{\columnsep}{4em}
  \begin{adjustwidth}{2em}{2em}
  \begin{multicols*}{2}}{\end{multicols*}
  \end{adjustwidth}
  }

\newlist{idxwordlist}{description}{3}
\setlist[idxwordlist, 1]{%
  itemsep=\baselineskip,
  labelindent=8em,
  font=\normalsize\bfseries,
}
\setlist[idxwordlist, 2]{%
  nosep,
  labelindent=2em,
  font=\wuhao\rm,
}
\setlist[idxwordlist, 3]{%
  nosep,
  labelindent=4em,
  font=\wuhao\rm,
}

\newcommand\bibstyle@numerical{\bibpunct{[}{]}{,}{s}{,}{\textsuperscript{,}}}
\citestyle{numerical}
\DeclareRobustCommand\inlinecite{\@inlinecite}
\def\@inlinecite#1{\begingroup\citestyle{inline}\let\@cite\NAT@citenum\citep{#1}\endgroup}
\let\onlinecite\inlinecite
\newcommand{\lcite}{\inlinecite}
\renewenvironment{thebibliography}[1]{%
  \ifheu@openright\cleardoublepage\else\clearpage\fi\phantomsection%
  %\heu@appendix@chapter*{\bibname}[\heu@bibname@etitle]
  \addcontentsline{toc}{chapter}{\bibname}
  \begin{center}
    ~\\
    \xiaoer\heiti\bibname
  \end{center}\par
  \normalsize
  \list{\@biblabel{\@arabic\c@enumiv}}%
  {\renewcommand{\makelabel}[1]{##1\hfill}
    \settowidth{\labelwidth}{\@biblabel{#1}}
    \setlength{\labelsep}{0.5em}
    \setlength{\itemindent}{0pt}
    \setlength{\leftmargin}{\labelsep+\labelwidth}
    \addtolength{\itemsep}{-0.8em}
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
  \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  \sloppy\frenchspacing
  \flushbottom
  \ifheu@splitbibitem
  \clubpenalty0
  \@clubpenalty \clubpenalty
  \widowpenalty0%
  \interlinepenalty-50%
  \else
  \clubpenalty4000
  \@clubpenalty \clubpenalty
  \widowpenalty4000%
  \interlinepenalty4000%
  \fi
\sfcode`\.\@m}
{\def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
\endlist\frenchspacing}
\patchcmd\NAT@citexnum{%
  \@ifnum{\NAT@ctype=\z@}{%
    \if*#2*\else\NAT@cmt#2\fi
  }{}%
  \NAT@mbox{\NAT@@close}%
}{%
  \NAT@mbox{\NAT@@close}%
  \@ifnum{\NAT@ctype=\z@}{%
    \if*#2*\else\textsuperscript{#2}\fi
  }{}%
}{}{}
\renewcommand\NAT@citesuper[3]{\ifNAT@swa
  \if*#2*\else#2\NAT@spacechar\fi
\unskip\kern\p@\textsuperscript{\NAT@@open#1\NAT@@close\if*#3*\else#3\fi}%
   \else #1\fi\endgroup}
\patchcmd{\NAT@citex}{%
  \if*#2*\else\NAT@cmt#2\fi
  \if\relax\NAT@date\relax\else\NAT@@close\fi
}{%
  \if\relax\NAT@date\relax\else\NAT@@close\fi
  \if*#2*\else\textsuperscript{#2}\fi
}{}{}
\renewcommand\NAT@cite%
    [3]{\ifNAT@swa\NAT@@open\if*#2*\else#2\NAT@spacechar\fi
        #1\NAT@@close\if*#3*\else\textsuperscript{#3}\fi\else#1\fi\endgroup}
\AtEndOfClass{\input{heuthesisbook.cfg}}
\AtEndOfClass{\sloppy}
\endinput
%%
%% End of file `examples/book/bachelor/heuthesisbook.cls'.
