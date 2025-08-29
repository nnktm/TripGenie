# TripGenie

AI を活用した旅行プラン生成アプリケーションです。

## 機能

- 🎯 目的地と期間を指定して旅行プランを自動生成
- 🌤️ 天気予報やイベント情報を自動検索
- 📸 Unsplash API を使用した観光地の美しい写真表示
- 📄 旅行プランを PDF として保存
- 🚇 交通手段や宿泊情報の詳細な調査
- 📱 レスポンシブデザインでモバイル対応

## セットアップ

### 1. 必要なパッケージのインストール

```bash
pip install flask openai python-dotenv requests reportlab
```

### 2. 環境変数の設定

`.env`ファイルを作成し、以下の API キーを設定してください：

```env
# OpenAI API設定
OPENAI_API_KEY=your_openai_api_key_here

# Tavily検索API設定
TAVILY_API_KEY=your_tavily_api_key_here

# Unsplash API設定
UNSPLASH_ACCESS_KEY=your_unsplash_access_key_here
UNSPLASH_SECRET_KEY=your_unsplash_secret_key_here
```

#### Unsplash API キーの取得方法

1. [Unsplash Developers](https://unsplash.com/developers)にアクセス
2. アカウントを作成またはログイン
3. 新しいアプリケーションを作成
4. Access Key と Secret Key を取得
5. `.env`ファイルに設定

### 3. アプリケーションの起動

```bash
python app.py
```

アプリケーションは `http://localhost:8080` で起動します。

## 使用方法

1. ホームページで目的地と旅行期間を入力
2. 出発地や宿泊日数などの詳細設定
3. 「旅行プランを作成」ボタンをクリック
4. AI が自動で旅行プランを生成
5. 観光地の写真と共に結果を表示
6. PDF としてダウンロード可能

## 新機能：観光地写真表示

- Unsplash API を使用して観光地の美しい写真を自動取得
- スケジュールに含まれる観光地の写真を表示
- 写真家のクレジット表示
- ホバーエフェクトとレスポンシブデザイン

## 注意事項

- Unsplash API の利用制限にご注意ください
- 写真は商用利用不可です（Unsplash ライセンスに従ってください）
- API キーは適切に管理し、公開リポジトリにアップロードしないでください

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。
