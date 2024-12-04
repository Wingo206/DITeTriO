using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class FakeReadOnlyList<T> : IReadOnlyList<T>
{
    private List<T> _internalList;

    // Constructors
    public FakeReadOnlyList()
    {
        _internalList = new List<T>();
    }

    public FakeReadOnlyList(IEnumerable<T> initialItems)
    {
        _internalList = new List<T>(initialItems);
    }

    // IReadOnlyList<T> implementation
    public T this[int index] => _internalList[index];
    public int Count => _internalList.Count;

    // Enumeration methods
    public IEnumerator<T> GetEnumerator() => _internalList.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    // Additional methods to modify the list
    public void Add(T item) => _internalList.Add(item);
    public void AddRange(IEnumerable<T> items) => _internalList.AddRange(items);
    public void Remove(T item) => _internalList.Remove(item);
    public void RemoveAt(int index) => _internalList.RemoveAt(index);
    public void Clear() => _internalList.Clear();

    // Optional: Conversion method to get the underlying list if needed
    public List<T> ToList() => new List<T>(_internalList);
}