const productsEl = document.querySelector("#products");
const template = document.querySelector("#product-template");
const form = document.querySelector("#review-form");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const summaryEl = document.querySelector("#summary");
const addButton = document.querySelector("#add-product");
const exampleSelect = document.querySelector("#example-select");
const loadExampleButton = document.querySelector("#load-example");

const examples = [
  {
    label: "Yoruba - pepper grill dinner",
    persona: {
      cultural_group: "Yoruba Nigerian diaspora",
      taste_profile: "I like pepper-forward food, grilled fish or meat, jollof rice, generous portions, lively ambience, and service that feels warm but efficient.",
      price_sensitivity: "medium",
      dietary_context: "No strict restriction, but I prefer a proper hot meal over dessert or coffee-only places.",
      dining_context: "Dinner after work with a moderate budget",
      review_style: "yoruba",
    },
    products: [{
      name: "Eko Pepper Grill",
      category: "Nigerian, Grill, Seafood, Rice Dishes",
      city: "Philadelphia",
      price_range: "2",
      summary: "A new casual Nigerian grill serving pepper soup, grilled fish, suya-style beef, jollof rice, and plantains in a lively room.",
    }],
  },
  {
    label: "Yoruba - dessert mismatch",
    persona: {
      cultural_group: "Yoruba Nigerian diaspora",
      taste_profile: "I want spicy, filling food with rice, grilled meat, seafood, strong flavor, and good value.",
      price_sensitivity: "high",
      dietary_context: "Avoids dessert-only and coffee-only places.",
      dining_context: "Lunch on a tight budget",
      review_style: "yoruba",
    },
    products: [{
      name: "Kiwi Frozen Yogurt",
      category: "Frozen Yogurt, Desserts",
      city: "Philadelphia",
      price_range: "2",
      summary: "A dessert shop focused on frozen yogurt, sweet toppings, smoothies, and light snacks.",
    }],
  },
  {
    label: "Yoruba - taco lunch",
    persona: {
      cultural_group: "Yoruba Nigerian diaspora",
      taste_profile: "I enjoy bold sauces, peppery food, rice bowls, grilled meat, quick service, and restaurants that give strong value for money.",
      price_sensitivity: "high",
      dietary_context: "No strict restriction.",
      dining_context: "Quick lunch between classes",
      review_style: "yoruba",
    },
    products: [{
      name: "Taco World",
      category: "Mexican, Tacos, Rice Bowls",
      city: "Atlanta",
      price_range: "1",
      summary: "A new counter-service taco spot with grilled chicken, spicy salsa, rice bowls, beans, and fast lunch portions.",
    }],
  },
  {
    label: "Igbo - hearty comfort dinner",
    persona: {
      cultural_group: "Igbo Nigerian diaspora",
      taste_profile: "I care about generous portions, filling food, meat or fish, rice dishes, practical value, and meals that feel satisfying after a long day.",
      price_sensitivity: "medium",
      dietary_context: "No strict restriction.",
      dining_context: "Dinner after work when I am hungry",
      review_style: "igbo",
    },
    products: [{
      name: "Coal City Kitchen",
      category: "West African, Grilled Meat, Rice, Stew",
      city: "Tampa",
      price_range: "2",
      summary: "A new casual restaurant serving smoky grilled goat, chicken stew, fried rice, jollof rice, and large plates.",
    }],
  },
  {
    label: "Igbo - BBQ value test",
    persona: {
      cultural_group: "Igbo Nigerian diaspora",
      taste_profile: "I like big portions, smoky meat, practical pricing, straightforward service, and food that feels worth the money.",
      price_sensitivity: "high",
      dietary_context: "No strict restriction.",
      dining_context: "Weekend lunch with friends",
      review_style: "igbo",
    },
    products: [{
      name: "Mission BBQ",
      category: "Barbeque, American, Sandwiches",
      city: "Philadelphia",
      price_range: "2",
      summary: "A barbecue restaurant with brisket, pulled chicken, ribs, sides, and a casual counter-service setup.",
    }],
  },
  {
    label: "Igbo - coffee weak fit",
    persona: {
      cultural_group: "Igbo Nigerian diaspora",
      taste_profile: "I want a full plate, rice or potatoes, meat, strong value, and enough food to feel properly satisfied.",
      price_sensitivity: "high",
      dietary_context: "Avoids snack-only places for meals.",
      dining_context: "Late breakfast when I need real food",
      review_style: "igbo",
    },
    products: [{
      name: "Passero's Coffee Roasters",
      category: "Coffee, Tea, Bakery",
      city: "Philadelphia",
      price_range: "2",
      summary: "A coffee shop focused on espresso drinks, pastries, small sandwiches, and a quiet work-friendly atmosphere.",
    }],
  },
  {
    label: "Hausa - halal family dinner",
    persona: {
      cultural_group: "Hausa Nigerian diaspora",
      taste_profile: "I prefer clean, calm, family-friendly places with rice dishes, grilled chicken, mild to medium spice, and reliable service.",
      price_sensitivity: "medium",
      dietary_context: "Halal-aware and avoids alcohol-heavy venues.",
      dining_context: "Family dinner",
      review_style: "hausa",
    },
    products: [{
      name: "Arewa Halal Grill",
      category: "Halal, Mediterranean, Grilled Chicken, Rice",
      city: "Philadelphia",
      price_range: "2",
      summary: "A new halal-friendly grill with chicken kebabs, rice platters, lentil soup, family seating, and no bar area.",
    }],
  },
  {
    label: "Hausa - alcohol risk",
    persona: {
      cultural_group: "Hausa Nigerian diaspora",
      taste_profile: "I value cleanliness, family suitability, rice dishes, grilled chicken, calm service, and clear halal-friendly options.",
      price_sensitivity: "medium",
      dietary_context: "Halal-aware and avoids alcohol-heavy venues.",
      dining_context: "Quiet dinner with family",
      review_style: "hausa",
    },
    products: [{
      name: "Village Whiskey",
      category: "Burgers, Whiskey Bar, American",
      city: "Philadelphia",
      price_range: "3",
      summary: "A lively bar restaurant known for whiskey, cocktails, burgers, and late-night dining.",
    }],
  },
  {
    label: "Hausa - rice house lunch",
    persona: {
      cultural_group: "Hausa Nigerian diaspora",
      taste_profile: "I like rice bowls, grilled chicken, clean service, moderate spice, halal-friendly menus, and restaurants that feel calm and practical.",
      price_sensitivity: "high",
      dietary_context: "Halal-aware.",
      dining_context: "Affordable lunch near campus",
      review_style: "hausa",
    },
    products: [{
      name: "Saffron Rice House",
      category: "Halal, Rice Bowls, Chicken, Middle Eastern",
      city: "Atlanta",
      price_range: "1",
      summary: "A new halal-friendly lunch spot with chicken rice bowls, chickpeas, grilled vegetables, and quick counter service.",
    }],
  },
  {
    label: "Balanced - sushi seafood",
    persona: {
      cultural_group: "Nigerian diaspora",
      taste_profile: "I like fresh seafood, clean flavors, good value, warm service, and a meal that feels carefully prepared without being too expensive.",
      price_sensitivity: "medium",
      dietary_context: "No strict restriction.",
      dining_context: "Casual dinner with one friend",
      review_style: "balanced",
    },
    products: [{
      name: "Matoi Sushi",
      category: "Sushi, Japanese, Seafood",
      city: "Philadelphia",
      price_range: "2",
      summary: "A sushi restaurant with fresh rolls, Korean dishes, seafood plates, quick service, and a casual dining room.",
    }],
  },
  {
    label: "Balanced - falafel value",
    persona: {
      cultural_group: "Nigerian diaspora",
      taste_profile: "I enjoy flavorful casual food, good value, filling portions, vegetarian options, rice or wraps, and a relaxed neighborhood feel.",
      price_sensitivity: "high",
      dietary_context: "No strict restriction.",
      dining_context: "Affordable late-night bite",
      review_style: "balanced",
    },
    products: [{
      name: "Bitar's",
      category: "Middle Eastern, Falafel, Sandwiches",
      city: "Philadelphia",
      price_range: "1",
      summary: "A casual Middle Eastern spot with falafel, shawarma, rice plates, fresh salads, and affordable portions.",
    }],
  },
];

function setStatus(text) {
  statusEl.textContent = text;
}

function setField(name, value) {
  const field = form.querySelector(`[name="${name}"]`);
  if (field) field.value = value;
}

function addProduct(defaults = {}) {
  const node = template.content.firstElementChild.cloneNode(true);
  Object.entries(defaults).forEach(([key, value]) => {
    const field = node.querySelector(`[data-field="${key}"]`);
    if (field) field.value = value;
  });
  node.querySelector(".remove-product").addEventListener("click", () => {
    if (productsEl.children.length > 1) node.remove();
  });
  productsEl.appendChild(node);
}

function resetProducts(products) {
  productsEl.innerHTML = "";
  products.forEach((product) => addProduct(product));
}

function loadExample(index) {
  const example = examples[index] || examples[0];
  Object.entries(example.persona).forEach(([key, value]) => setField(key, value));
  resetProducts(example.products);
  summaryEl.textContent = "Example loaded. Run the deployed LightGCN reranker to generate ratings and reviews.";
  resultsEl.innerHTML = "";
  setStatus("Ready");
}

function productFromCard(card) {
  const value = (field) => {
    const input = card.querySelector(`[data-field="${field}"]`);
    return input ? input.value.trim() : "";
  };
  return {
    name: value("name"),
    category: value("category"),
    city: value("city"),
    price_range: value("price_range"),
    rating: null,
    review_count: null,
    summary: value("summary"),
    image_url: value("image_url"),
  };
}

function requestFromForm() {
  const data = new FormData(form);
  return {
    persona: {
      cultural_group: data.get("cultural_group"),
      taste_profile: data.get("taste_profile"),
      price_sensitivity: data.get("price_sensitivity"),
      dietary_context: data.get("dietary_context"),
      dining_context: data.get("dining_context"),
      review_style: data.get("review_style"),
    },
    products: Array.from(productsEl.children).map(productFromCard),
    provider: "lightgcn",
    max_tokens: 900,
  };
}

function renderReviews(payload) {
  resultsEl.innerHTML = "";
  summaryEl.textContent = payload.overall_summary;
  if (payload.warnings && payload.warnings.length) {
    payload.warnings.forEach((warning) => {
      const div = document.createElement("div");
      div.className = "summary warning";
      div.textContent = warning;
      resultsEl.appendChild(div);
    });
  }

  payload.reviews.forEach((review) => {
    const card = document.createElement("article");
    card.className = `review-card ${review.sentiment}`;
    const reasons = (review.reasons || []).filter(Boolean);
    card.innerHTML = `
      <div class="review-head">
        <div>
          <h3>${escapeHtml(review.name)}</h3>
          <div class="sentiment">${escapeHtml(review.sentiment)}</div>
        </div>
        <div class="rating">${Number(review.rating).toFixed(1)}/5</div>
      </div>
      <p class="review-text">${escapeHtml(review.review)}</p>
      ${reasons.length ? `
        <details class="evidence">
          <summary>Model evidence</summary>
          <ul>
            ${reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}
          </ul>
        </details>
      ` : ""}
    `;
    resultsEl.appendChild(card);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Running");
  const body = requestFromForm();
  try {
    const response = await fetch("/api/generate-review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    renderReviews(await response.json());
    setStatus("Done");
  } catch (error) {
    summaryEl.textContent = "Generation failed.";
    resultsEl.innerHTML = `<div class="summary warning">${escapeHtml(error.message)}</div>`;
    setStatus("Error");
  }
});

addButton.addEventListener("click", () => {
  addProduct({
    name: "Lolis Mexican Cravings",
    category: "Mexican, Restaurants, Specialty Food",
    city: "Tampa",
    price_range: "1",
    summary: "A casual Mexican spot with bold flavors, quick service, and strong value.",
  });
});

if (!exampleSelect.children.length) {
  examples.forEach((example, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = example.label;
    exampleSelect.appendChild(option);
  });
}

loadExampleButton.addEventListener("click", () => {
  loadExample(Number(exampleSelect.value));
});

exampleSelect.addEventListener("change", () => {
  loadExample(Number(exampleSelect.value));
});

loadExample(0);
