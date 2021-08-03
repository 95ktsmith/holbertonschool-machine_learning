-- Creates a trigger that decreases the quantity of an item after adding order
CREATE TRIGGER decrease_items
    AFTER INSERT ON orders FOR EACH ROW
    UPDATE
        items
    SET
        quantity = quantity - new.number
    WHERE
        items.name = new.item_name;
