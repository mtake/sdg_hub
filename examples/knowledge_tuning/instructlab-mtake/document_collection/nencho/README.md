# [令和6年分 年末調整のしかた](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/01.htm)

## [利用規約](https://www.nta.go.jp/chuijiko/copy.htm)
著作権は、特記されていない限り国税庁に帰属し、権利表記の記載がない限り「[公共データ利用規約（第1.0版）](https://www.digital.go.jp/resources/open_data/public_data_license_v1.0)」に準拠した利用条件の下で、利用することができます。公共データ利用規約（第1.0版）のうち、本サイト独自の出典記載例や本利用ルールの適用を受けないコンテンツ等のサイトによって内容が異なる部分の情報については、下記、「[公共データ利用規約（第1.0版）に関する重要情報](#公共データ利用規約第10版に関する重要情報)」を参照してください。

## 公共データ利用規約（第1.0版）に関する重要情報
1.  出典の記載について
    1.  コンテンツを利用する際は出典を記載してください。出典の記載方法は以下のとおりです。
    ```
    （出典記載例）出典：国税庁ホームページ（当該ぺージのURL）
    ```
    2.  コンテンツを編集・加工等して利用する場合は、上記出典とは別に、編集・加工等を行ったことを記載してください。なお、編集・加工等した情報を、あたかも国（又は府省等）が作成したかのような態様で公表・利用してはいけません。
    ```
    （コンテンツを編集・加工等して利用する場合の記載例）「○○調査結果」（国税庁）（当該ページの URL）を加工して作成
    ```
（以下省略）

<!--
## 一括ダウンロード
- [一括ダウンロード](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/nencho_all.pdf)
-->

## 項目別ダウンロード
- [表紙・目次 p.1-2](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/01.pdf)
- [I 昨年と比べて変わった点(定額減税) p.3-4](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/02.pdf)
- [II 年末調整とは p.5-6](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/03.pdf)
- [III 年末調整のしかた、1 年末調整の手順 p.7](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/05.pdf)
- [2 各種控除額の確認、2-1 扶養控除等(異動)申告書の受理と内容の確認 p.8-16(重複あり)](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/07.pdf)
- [2-2 基礎控除申告書、配偶者控除等(兼定額減税)申告書及び所得金額調整控除申告書の受理と内容の確認 p.16(重複あり)-21(重複あり)](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/08.pdf)
- [2-3 保険料控除申告書の受理と内容の確認 p.21(重複あり)-28](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/09.pdf)
- [2-4 (特定増改築等)住宅借入金等特別控除申告書の受理と内容の確認 p.29-33](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/10.pdf)
- [3 年税額の計算 p.34-38](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/11.pdf)
- [4 過不足額の精算 p.39-46](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/12.pdf)
  - (使用しない) [見開きの図 p.42-43](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/12-2.pdf)
- [5 税額の納付と所得税徴収高計算書(納付書)の記載 p.47](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/13.pdf)
- [6 年末調整後に給与の追加払や扶養親族等の異動があった場合の再調整 p.48](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/14.pdf)
- [IV 令和7年分の給与の源泉徴収事務 p.49-50](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/15.pdf)
- [令和6年分の年末調整等のための給与所得控除後の給与等の金額の表 p.51-59](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/17.pdf)
- [令和6年分の年末調整のための算出所得税額の速算表、令和6年分の配偶者控除額及び配偶者特別控除額の一覧表、令和6年分の基礎控除額の表、令和6年分の扶養控除額等の表 p.60-61](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/18.pdf)
- [「令和6年分の扶養控除額及び障害者等の控除額の合計額の早見表」(64ページ)の使い方 p.62-63](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/19.pdf)
- [令和6年分の扶養控除額及び障害者等の控除額の合計額の早見表 p.64](https://www.nta.go.jp/publication/pamph/gensen/nencho2024/pdf/20.pdf)

## 項目別pdfをmarkdownへ変換後の手作業での修正
```bash
sed -i 's/^G//g' ??.md  # ^G (^V^Gで入力) を削除
sed -i 's/^Z//g' ??.md  # ^Z (^V^Zで入力) を削除
sed -i 's/　/ /g' ??.md  # 全角スペースを半角スペースに変換
```
- 誤認識したテキストを削除する
- 余分な空白、空行を削除する
- レイアウトを修正する
- 図を削除する
- 表を修正する
- 数字を統一？

<!--
⑴
⑵
⑶
⑷
⑸
⑹
⑺
⑻
⑼
⑽
-->

## 項目別markdownの結合
```bash
rm nencho.md; for f in ??.md; do cat $f >> nencho.md; done
```
