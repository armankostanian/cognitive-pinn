# Инструкция: как залить проект на GitHub

> Здесь **три варианта** — выберите тот, что вам ближе:
>
> **🟢 Вариант 1 (Easy):** через web-интерфейс GitHub. Подходит для **первого раза**, без установки чего-либо. Минусы: не очень удобно для будущих обновлений.
>
> **🟡 Вариант 2 (Recommended):** через GitHub Desktop — графический клиент. Лучший компромисс для аспиранта: визуальный интерфейс, не нужно знать команды git.
>
> **🔴 Вариант 3 (Pro):** через командную строку. Для тех, кто умеет в терминал.

---

## Зачем это вообще нужно

Журнал Chaos, Solitons &amp; Fractals требует поле **«Code availability statement»** в submission. У нас в статье уже есть:

```
The reference implementation, the synthetic identifiability suite, and
the trained Cognitive-PINN models are available at
https://github.com/[anon]/cognitive-pinn upon publication.
```

Эта ссылка должна **реально работать** перед submission. Без рабочего GitHub-репозитория статью завернут на desk-review. Эта инструкция — чтобы ссылка заработала за 15–30 минут.

---

## Шаг 0 (для всех вариантов): создать GitHub-аккаунт и репозиторий

### 0.1. Создать аккаунт

Если уже есть — пропустите.

1. Откройте https://github.com
2. Нажмите **Sign up** в правом верхнем углу.
3. Введите email, пароль, username (это будет в URL вашего репозитория, например `https://github.com/your-name/cognitive-pinn`).
4. Подтвердите email.

> 💡 **Совет**: используйте университетский email — у вас будут бесплатные приватные репозитории и доступ к GitHub Education. Зарегистрируйтесь как студент: https://education.github.com/pack

### 0.2. Создать репозиторий

1. На https://github.com нажмите **+** в правом верхнем углу → **New repository**.
2. Заполните:
   - **Repository name**: `cognitive-pinn`
   - **Description**: `Physics-Informed Modeling of Student Knowledge Dynamics with Identifiable Parameters and On-Device Adaptation`
   - **Public** (чтобы рецензенты могли смотреть; иначе придётся раздавать приглашения)
   - ⛔ **НЕ ставьте** галочки «Add README», «.gitignore», «Add license» — мы зальём свои файлы.
3. Нажмите **Create repository**.

После создания у вас будет **пустой репозиторий**. URL: `https://github.com/YOUR_USERNAME/cognitive-pinn`

### 0.3. Подготовить файлы

Распакуйте `cognitive-pinn-template.zip` в удобную папку, например:
- Mac: `~/Documents/cognitive-pinn/`
- Windows: `C:\Users\YourName\Documents\cognitive-pinn\`

Внутри должно быть:

```
cognitive-pinn/
├── README.md
├── HOW_TO_RUN.md
├── HOW_TO_GITHUB.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── cpinn_assist09_colab.ipynb
├── src/
│   ├── __init__.py
│   ├── cpinn_model.py
│   ├── cpinn_assist09_loader.py
│   ├── cpinn_synthetic_pipeline.py
│   ├── cpinn_identifiability_K2.py
│   └── cpinn_oracle_check.py
├── figures/
│   └── (5 PNG-файлов)
└── docs/
    └── (3 MD-файла)
```

---

## 🟢 Вариант 1: Web-интерфейс GitHub (самый простой, без установки)

Подходит для **первой заливки**. Минус — каждый раз для обновления нужно перетаскивать файлы заново.

### 1.1. Открыть пустой репозиторий

Перейдите на `https://github.com/YOUR_USERNAME/cognitive-pinn`

Вы увидите страницу с надписью «Quick setup».

### 1.2. Загрузить файлы

Найдите ссылку **«uploading an existing file»** — она в синей подсказке внизу страницы.

Откроется страница с большим серым прямоугольником: «Drag files here to add them to your repository».

### 1.3. Перетащить ВСЕ файлы

⚠️ **Очень важно**: GitHub web-интерфейс **не поддерживает** перетаскивание целых папок. Только файлы.

Поэтому действуйте так:

**Шаг 1.3a.** Перетащите все файлы и папки из корня:
- `README.md`
- `HOW_TO_RUN.md`
- `HOW_TO_GITHUB.md`
- `LICENSE`
- `requirements.txt`
- `.gitignore`
- `cpinn_assist09_colab.ipynb`

И **папки** `src/`, `figures/`, `docs/` — Chrome их перетаскивает целиком (Safari — нет).

> 💡 Если ваш браузер (особенно Safari) **не принимает папки** — раскройте каждую папку и перетаскивайте файлы по одному, **указывая путь** в специальном поле (см. шаг 1.3b).

**Шаг 1.3b.** Альтернатива для Safari / упрямых браузеров: для каждого файла из подпапки введите путь вручную.
- В поле «Type a name to add an empty directory» напишите, например, `src/cpinn_model.py`. Это создаст папку `src/` и положит файл туда.
- Содержимое файла нужно будет вставить руками. Поэтому проще использовать Chrome.

### 1.4. Сообщение коммита

Внизу страницы есть форма «Commit changes»:

- **Commit message**: `Initial commit: Cognitive-PINN repository`
- **Description**: оставьте пустым
- Выберите радиокнопку **Commit directly to the `main` branch**

Нажмите **Commit changes**.

### 1.5. Проверить результат

Через несколько секунд страница перезагрузится, и вы увидите все ваши файлы. README автоматически отрендерится снизу.

🎉 **Готово!** Репозиторий онлайн.

### 1.6. Как обновлять файлы в Web-интерфейсе

Когда нужно обновить, например, `cognitive_pinn_P1.tex` после Colab-прогона:

1. Откройте файл на GitHub (кликните на него в списке).
2. Нажмите на иконку **карандаша** ✏️ в правом верхнем углу.
3. Отредактируйте онлайн или **загрузите новую версию** через «Upload files» в верхнем меню.
4. Снизу страницы — Commit message типа `Update Table 3 with ASSIST-09 results`, и Commit changes.

---

## 🟡 Вариант 2: GitHub Desktop (рекомендую для аспиранта)

GitHub Desktop — **графический клиент** для git. Не нужно знать команды. Установка 5 минут, потом всё в визуальном интерфейсе.

### 2.1. Установить GitHub Desktop

Скачайте с https://desktop.github.com/

- **Mac**: dmg-файл, перетащите GitHub Desktop в Applications.
- **Windows**: exe-инсталлер, нажмите «Install» и подождите.

### 2.2. Войти в аккаунт

Откройте GitHub Desktop. Появится экран приветствия:
- **Sign in to GitHub.com**.

Откроется браузер для авторизации. После согласия — вернитесь в GitHub Desktop.

### 2.3. Связать с репозиторием

В GitHub Desktop меню:
- **File** → **Clone repository...**

Вкладка **GitHub.com** покажет список ваших репозиториев. Выберите `cognitive-pinn` и:
- **Local path**: укажите папку, где будет копия (например, `~/Documents/cognitive-pinn-clone/`)
- **Clone**.

> ⚠️ Это создаст **пустую папку** (так как репозиторий ещё пустой). Не путайте с папкой, где лежат ваши файлы из архива.

### 2.4. Скопировать ваши файлы в эту папку

В Finder/Explorer откройте две папки рядом:
- Слева: распакованный `cognitive-pinn-template/` с вашими файлами
- Справа: только что созданная `cognitive-pinn-clone/` (она пустая, кроме скрытой `.git/`)

**Скопируйте ВСЕ файлы и папки** из левой в правую (Cmd+A → Cmd+C на Mac или Ctrl+A → Ctrl+C на Windows, потом вставка в правую папку).

### 2.5. GitHub Desktop увидел изменения

Вернитесь в GitHub Desktop. Слева вы увидите длинный список файлов с галочками — это все новые файлы, которые ждут commit.

Внизу слева:
- **Summary (required)**: `Initial commit: Cognitive-PINN repository`
- **Description**: оставьте пустым.

Нажмите **Commit to main** (синяя кнопка).

### 2.6. Push на GitHub

После commit GitHub Desktop покажет в верхней панели:
**Push origin** (с индикатором сколько коммитов ждёт отправки).

Нажмите **Push origin**.

Через 10–30 секунд (зависит от размера) — всё на GitHub.

### 2.7. Проверить

Откройте `https://github.com/YOUR_USERNAME/cognitive-pinn` в браузере. Должны увидеть все файлы и красивый README.

🎉 **Готово!**

### 2.8. Как обновлять с GitHub Desktop

Каждый раз, когда нужно обновить:

1. Отредактируйте файл в любом редакторе (например, `cognitive_pinn_P1.tex`).
2. Откройте GitHub Desktop — он автоматически увидит изменения.
3. Внизу слева напишите Summary, например: `Update Table 3 with ASSIST-09 results`.
4. **Commit to main** → **Push origin**.

Это очень удобно: 3 клика и обновление онлайн.

---

## 🔴 Вариант 3: Командная строка (для тех, кто умеет в терминал)

Подходит, если у вас уже стоит git и вы знаете базовые команды.

### 3.1. Установить git (если не установлен)

**Mac**: откройте Terminal и введите `git --version`. Если git нет — система предложит установить (соглашайтесь).

**Windows**: скачайте с https://git-scm.com/download/win, инсталлер с дефолтными настройками. Откройте «Git Bash».

**Linux**: `sudo apt install git`

### 3.2. Создать Personal Access Token

С 2021 GitHub не разрешает push через пароль. Нужен токен.

1. На GitHub: ваш аватар → **Settings** → **Developer settings** (внизу левого меню) → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**.
2. Заполните:
   - **Note**: `cognitive-pinn`
   - **Expiration**: 90 days
   - **Scopes**: галочка `repo`
3. **Generate token**.
4. **СКОПИРУЙТЕ ТОКЕН СРАЗУ** — он показывается только один раз. Сохраните в надёжное место.

Токен выглядит как `ghp_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890`.

### 3.3. Настроить git (один раз на компьютер)

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@university.edu"
```

### 3.4. Залить проект

```bash
cd ~/Documents/cognitive-pinn   # или где у вас лежит распакованный архив

git init
git branch -M main
git add .
git commit -m "Initial commit: Cognitive-PINN repository"
git remote add origin https://github.com/YOUR_USERNAME/cognitive-pinn.git
git push -u origin main
```

При `git push` git спросит:
- **Username**: ваш GitHub username
- **Password**: введите **Personal Access Token** (НЕ пароль от GitHub!)

После пары секунд:
```
* [new branch]      main -> main
```

🎉 **Готово!** Откройте `https://github.com/YOUR_USERNAME/cognitive-pinn` для проверки.

### 3.5. Обновлять файлы

```bash
# (отредактировали файл в редакторе)
git add cpinn_model.py
git commit -m "Update model"
git push
```

---

## Шаг 99 (для всех): обновить ссылку в статье

После заливки откройте `cognitive_pinn_P1.tex`, найдите:

```latex
\url{https://github.com/[anon]/cognitive-pinn}
```

И замените на:
```latex
\url{https://github.com/YOUR_USERNAME/cognitive-pinn}
```

> ⚠️ **Внимание**: Уточните политику рецензирования целевого журнала (Chaos, Solitons &amp; Fractals — single-blind). Если требуется анонимность, используйте `[anon]` в URL и анонимный архив через https://anonymous.4open.science/.

---

## Дополнительно: создать Release для submission

Когда статья готова к submission, имеет смысл сделать **release** — это «снимок» кода на конкретный момент. Удобно для воспроизводимости (рецензент может проверить именно ту версию, что упоминается в статье).

### Через web-интерфейс

1. На странице репозитория → **Releases** (справа).
2. **Create a new release**.
3. **Choose a tag**: введите `v1.0-submission` и кликните «Create new tag».
4. **Release title**: `Cognitive-PINN: KBS submission`.
5. **Description**:
   ```
   Initial public release accompanying the submission to
   in preparation, 2026.

   Includes:
   - PyTorch reference implementation
   - Colab notebook for ASSISTments-2009 reproduction
   - Synthetic identifiability validation suite
   - Full proofs in docs/
   ```
6. **Publish release**.

В статье тогда лучше ссылаться на конкретный tag:
```latex
\url{https://github.com/YOUR_USERNAME/cognitive-pinn/releases/tag/v1.0-submission}
```

Это **best practice** для воспроизводимости.

---

## Решение проблем

### «Authentication failed» при push
1. Возможно, вы ввели пароль, а не Token. Введите токен из шага 3.2.
2. Если токен правильный, но всё равно ошибка — токен мог истечь. Создайте новый.

### Файл больше 100 MB
GitHub не пускает большие файлы. Скорее всего это датасет или PDF-сборка. Добавьте в `.gitignore` и сделайте:
```bash
git rm --cached big_file.csv
git commit -m "Remove large file"
git push
```

В нашем `.gitignore` уже исключены типичные большие файлы: `data/`, `*.csv`, `artifacts/`.

### Случайно залил Personal Access Token в код
1. **Срочно отзовите токен** на GitHub (Settings → Tokens → Revoke).
2. Создайте новый.
3. Удалите токен из кода и сделайте новый commit.

### `remote origin already exists`
Вы случайно запустили `git remote add` дважды:
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/cognitive-pinn.git
```

### Github Desktop не видит изменения файлов
1. **Repository** → **Refresh** (или Cmd+R).
2. Если не помогло — закройте и откройте Desktop заново.

### Web-интерфейс: «File too large to upload»
GitHub web позволяет файлы до **25 MB**. Для больших — используйте GitHub Desktop или CLI. Но проверьте — нет ли в проекте чего-то лишнего (PDF статьи, ZIP-архив, кеш).

---

## Чек-лист

После всех шагов:
- [ ] Репозиторий открывается в браузере
- [ ] README красиво отображается на главной
- [ ] `cpinn_assist09_colab.ipynb` открывается с preview
- [ ] Папки `src/`, `figures/`, `docs/` все на месте
- [ ] Картинки в `figures/` показываются
- [ ] Ссылка в статье `cognitive_pinn_P1.tex` обновлена
- [ ] При желании — Release `v1.0-submission` создан

🎉 Если все ✅ — у вас рабочий публичный репозиторий, готовый для submission!

---

## Что писать в Cover Letter про код

Когда будете подавать в журнал, в cover letter включите:

> «The complete source code, including a Colab notebook reproducing all
> reported results, is publicly available at
> https://github.com/YOUR_USERNAME/cognitive-pinn (release `v1.0-submission`).
> The repository contains the PyTorch reference implementation, the
> synthetic identifiability validation suite, the data loader for
> ASSISTments-2009 (DKVMN preprocessed), and full mathematical proofs
> of the theorems stated in the paper.»

Это сильный плюс для редактора и рецензентов — показывает, что вы серьёзны к воспроизводимости.

---

## Какой вариант выбрать?

| Вы | Рекомендую |
|---|---|
| Никогда не работал с git, нужно «один раз залить» | Вариант 1 (Web) |
| Аспирант, буду обновлять статью и код регулярно | **Вариант 2 (Desktop)** |
| Уверенно работаю в терминале | Вариант 3 (CLI) |
| Использую VS Code | Вариант 3, но в VS Code встроенный git тоже хорош |

Для большинства научных подач **GitHub Desktop** — оптимум. Один раз настроил, дальше 3 клика на каждое обновление.

---

## После заливки — следующий шаг

Скачайте архив `cpinn_artifacts.zip` с результатами Colab-прогона и **загрузите его в Releases как asset** (если у вас есть release). Это даст рецензентам прямую возможность скачать ваши итоговые числа без необходимости перепрогонять Colab.

Удачи!
