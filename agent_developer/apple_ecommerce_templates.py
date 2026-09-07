"""
Modèles et Templates E-Commerce Style Apple (Design Épuré, Typographie Moderne, Glassmorphism, Panier Interactif).
Permet à Sarah Ngin de générer des sites Web complets prêts à l'emploi.
"""

APPLE_ECOMMERCE_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sarah Store • Design Style Apple</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #000000;
            --card-bg: rgba(22, 22, 23, 0.8);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent-blue: #2997ff;
            --accent-glow: rgba(41, 151, 255, 0.35);
            --text-main: #f5f5f7;
            --text-sub: #86868b;
            --radius-apple: 22px;
            --transition-smooth: all 0.35s cubic-bezier(0.25, 0.1, 0.25, 1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            overflow-x: hidden;
            line-height: 1.5;
        }

        /* NAVIGATION STYLE APPLE */
        nav {
            position: fixed;
            top: 0;
            width: 100%;
            height: 52px;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: saturate(180%) blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 40px;
            z-index: 1000;
        }

        .nav-brand {
            font-weight: 700;
            font-size: 18px;
            letter-spacing: -0.5px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .nav-brand span {
            background: linear-gradient(135deg, #2997ff, #9b51e0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-links {
            display: flex;
            gap: 32px;
            list-style: none;
            font-size: 13px;
            color: var(--text-sub);
        }

        .nav-links a {
            color: inherit;
            text-decoration: none;
            transition: color 0.2s;
        }

        .nav-links a:hover {
            color: #fff;
        }

        .cart-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #fff;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition-smooth);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .cart-btn:hover {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
        }

        /* HERO SECTION */
        .hero {
            padding: 140px 20px 80px;
            text-align: center;
            max-width: 1000px;
            margin: 0 auto;
        }

        .hero-tag {
            font-size: 14px;
            font-weight: 600;
            color: #f56300;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .hero h1 {
            font-size: 64px;
            font-weight: 700;
            letter-spacing: -1.5px;
            line-height: 1.08;
            margin-bottom: 16px;
            background: linear-gradient(180deg, #ffffff 0%, #a1a1a6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            font-size: 21px;
            color: var(--text-sub);
            max-width: 650px;
            margin: 0 auto 32px;
        }

        .hero-actions {
            display: flex;
            justify-content: center;
            gap: 16px;
        }

        .btn-primary {
            background: var(--accent-blue);
            color: #fff;
            padding: 12px 26px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 14px;
            border: none;
            cursor: pointer;
            transition: var(--transition-smooth);
            text-decoration: none;
        }

        .btn-primary:hover {
            box-shadow: 0 0 25px var(--accent-glow);
            transform: scale(1.02);
        }

        .btn-secondary {
            background: transparent;
            color: var(--accent-blue);
            padding: 12px 26px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 14px;
            border: 1px solid rgba(41, 151, 255, 0.4);
            cursor: pointer;
            transition: var(--transition-smooth);
        }

        .btn-secondary:hover {
            background: rgba(41, 151, 255, 0.1);
        }

        /* GRILLE DE PRODUITS STYLE APPLE */
        .products-section {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px 100px;
        }

        .section-header {
            margin-bottom: 40px;
        }

        .section-header h2 {
            font-size: 36px;
            font-weight: 700;
            letter-spacing: -0.8px;
        }

        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 28px;
        }

        .product-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: var(--radius-apple);
            padding: 36px 30px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 440px;
            transition: var(--transition-smooth);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(20px);
        }

        .product-card:hover {
            border-color: rgba(255, 255, 255, 0.3);
            transform: translateY(-6px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
        }

        .product-tag {
            font-size: 12px;
            color: #f56300;
            font-weight: 600;
            text-transform: uppercase;
        }

        .product-title {
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 6px 0 8px;
        }

        .product-desc {
            font-size: 14px;
            color: var(--text-sub);
            margin-bottom: 24px;
        }

        .product-preview {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.05) 0%, transparent 70%);
            border-radius: 16px;
            margin-bottom: 24px;
            font-size: 64px;
        }

        .product-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 18px;
        }

        .product-price {
            font-size: 18px;
            font-weight: 700;
            color: #fff;
        }

        .btn-add {
            background: #fff;
            color: #000;
            border: none;
            padding: 8px 18px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition-smooth);
        }

        .btn-add:hover {
            background: var(--accent-blue);
            color: #fff;
        }

        /* TIROIR PANIER (CART DRAWER) */
        .cart-drawer {
            position: fixed;
            top: 0;
            right: -420px;
            width: 400px;
            height: 100vh;
            background: rgba(18, 18, 20, 0.95);
            backdrop-filter: blur(25px);
            border-left: 1px solid var(--card-border);
            z-index: 2000;
            padding: 30px;
            display: flex;
            flex-direction: column;
            transition: right 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: -20px 0 50px rgba(0, 0, 0, 0.8);
        }

        .cart-drawer.open {
            right: 0;
        }

        .cart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .cart-items {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .cart-item {
            background: rgba(255, 255, 255, 0.04);
            border-radius: 12px;
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .cart-total-box {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 20px;
            margin-top: 20px;
        }

        .cart-total-row {
            display: flex;
            justify-content: space-between;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 16px;
        }

        .btn-checkout {
            width: 100%;
            background: var(--accent-blue);
            color: #fff;
            padding: 14px;
            border-radius: 14px;
            font-weight: 700;
            border: none;
            cursor: pointer;
            font-size: 15px;
            transition: var(--transition-smooth);
        }

        .btn-checkout:hover {
            box-shadow: 0 0 25px var(--accent-glow);
        }
    </style>
</head>
<body>
    <nav>
        <div class="nav-brand">Sarah Store <span>• Apple Style</span></div>
        <ul class="nav-links">
            <li><a href="#products">Produits</a></li>
            <li><a href="#specs">Spécifications</a></li>
            <li><a href="#ai">Sarah Ngin AI</a></li>
        </ul>
        <button class="cart-btn" onclick="toggleCart()">
            🛒 Panier (<span id="cartBadge">0</span>)
        </button>
    </nav>

    <header class="hero">
        <div class="hero-tag">Nouveauté 2026</div>
        <h1>L'intelligence pure.<br>Conçue pour votre poche.</h1>
        <p>Découvrez la gamme d'appareils optimisés pour faire tourner le modèle Sarah Ngin en local sur 4 Go de RAM.</p>
        <div class="hero-actions">
            <a href="#products" class="btn-primary">Acheter</a>
            <button class="btn-secondary" onclick="alert('Sarah Ngin fonctionne à 100% de zéro sans serveur externe.')">En savoir plus</button>
        </div>
    </header>

    <section class="products-section" id="products">
        <div class="section-header">
            <h2>Nos modèles phares.</h2>
        </div>
        <div class="product-grid">
            <!-- Produit 1 -->
            <div class="product-card">
                <div>
                    <div class="product-tag">Ultra-Léger</div>
                    <h3 class="product-title">Sarah Ngin Mobile</h3>
                    <p class="product-desc">Calibré pour iPhone 14 avec 4 Go de RAM. Inférence INT8 instantanée.</p>
                </div>
                <div class="product-preview">📱</div>
                <div class="product-footer">
                    <span class="product-price">799 €</span>
                    <button class="btn-add" onclick="addToCart('Sarah Ngin Mobile (iPhone 14 Edition)', 799)">Ajouter</button>
                </div>
            </div>

            <!-- Produit 2 -->
            <div class="product-card">
                <div>
                    <div class="product-tag">Pro Performance</div>
                    <h3 class="product-title">Sarah Studio Pro</h3>
                    <p class="product-desc">Modèle 60M paramètres avec moteur de recherche web et mémoire infinie.</p>
                </div>
                <div class="product-preview">💻</div>
                <div class="product-footer">
                    <span class="product-price">1 299 €</span>
                    <button class="btn-add" onclick="addToCart('Sarah Studio Pro 60M', 1299)">Ajouter</button>
                </div>
            </div>

            <!-- Produit 3 -->
            <div class="product-card">
                <div>
                    <div class="product-tag">Édition Spéciale</div>
                    <h3 class="product-title">Sarah Ngin Multilingual</h3>
                    <p class="product-desc">Dictionnaire Hébreu-Français et Littérature intégrée dans les poids neuronaux.</p>
                </div>
                <div class="product-preview">🌐</div>
                <div class="product-footer">
                    <span class="product-price">999 €</span>
                    <button class="btn-add" onclick="addToCart('Sarah Ngin Multilingual Edition', 999)">Ajouter</button>
                </div>
            </div>
        </div>
    </section>

    <!-- TIROIR PANIER -->
    <div class="cart-drawer" id="cartDrawer">
        <div class="cart-header">
            <h3>Votre Panier</h3>
            <button onclick="toggleCart()" style="background:none; border:none; color:#fff; font-size:20px; cursor:pointer;">✕</button>
        </div>
        <div class="cart-items" id="cartItems">
            <p style="color:var(--text-sub); text-align:center; margin-top:40px;">Votre panier est vide.</p>
        </div>
        <div class="cart-total-box">
            <div class="cart-total-row">
                <span>Total :</span>
                <span id="cartTotal">0 €</span>
            </div>
            <button class="btn-checkout" onclick="checkout()">Commander avec Apple Pay</button>
        </div>
    </div>

    <script>
        let cart = [];

        function toggleCart() {
            document.getElementById("cartDrawer").classList.toggle("open");
        }

        function addToCart(name, price) {
            cart.push({ name, price, id: Date.now() });
            updateCartUI();
            toggleCart();
        }

        function updateCartUI() {
            document.getElementById("cartBadge").innerText = cart.length;
            const container = document.getElementById("cartItems");
            
            if (cart.length === 0) {
                container.innerHTML = '<p style="color:var(--text-sub); text-align:center; margin-top:40px;">Votre panier est vide.</p>';
                document.getElementById("cartTotal").innerText = "0 €";
                return;
            }

            container.innerHTML = "";
            let total = 0;
            cart.forEach((item, index) => {
                total += item.price;
                const div = document.createElement("div");
                div.className = "cart-item";
                div.innerHTML = `
                    <div>
                        <div style="font-weight:600; font-size:13px;">${item.name}</div>
                        <div style="font-size:12px; color:var(--accent-blue);">${item.price} €</div>
                    </div>
                    <button onclick="removeItem(${index})" style="background:none; border:none; color:#ff453a; cursor:pointer; font-size:12px;">Supprimer</button>
                `;
                container.appendChild(div);
            });

            document.getElementById("cartTotal").innerText = total + " €";
        }

        function removeItem(index) {
            cart.splice(index, 1);
            updateCartUI();
        }

        function checkout() {
            if (cart.length === 0) return alert("Votre panier est vide !");
            alert("✅ Commande validée avec succès ! Total : " + document.getElementById("cartTotal").innerText);
            cart = [];
            updateCartUI();
            toggleCart();
        }
    </script>
</body>
</html>
"""

def get_apple_ecommerce_html() -> str:
    return APPLE_ECOMMERCE_HTML
