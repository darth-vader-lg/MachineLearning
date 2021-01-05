using MachineLearning;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML
{
   /// <summary>
   /// Estensioni per l'accesso ai gestori di dati
   /// </summary>
   public static class DataViewExtensions
   {
      #region Methods
      /// <summary>
      /// Restituisce il valore float di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="column">Indice colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static float GetFloat(this IDataView dataView, int column, int row = 0) => dataView.GetValue<float>(column, row);
      /// <summary>
      /// Restituisce il valore float di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="columnName">Nome colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static float GetFloat(this IDataView dataView, string columnName, int row = 0) => dataView.GetValue<float>(columnName, row);
      /// <summary>
      /// Restituisce il valore stringa di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="column">Indice colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static string GetString(this IDataView dataView, int column, int row = 0) => dataView.GetValue<ReadOnlyMemory<char>>(column, row).ToString();
      /// <summary>
      /// Restituisce il valore stringa di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="columnName">Nome colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static string GetString(this IDataView dataView, string columnName, int row = 0) => dataView.GetValue<ReadOnlyMemory<char>>(columnName, row).ToString();
      /// <summary>
      /// Restituisce il valore di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="column">Indice colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static T GetValue<T>(this IDataView dataView, int column, int row = 0)
      {
         var cursor = dataView.GetRowCursor(new[] { dataView.Schema[column] });
         while (cursor.MoveNext()) {
            if (cursor.Position == row) {
               var val = default(T);
               cursor.GetGetter<T>(dataView.Schema[column]).Invoke(ref val);
               return val;
            }
         }
         return default;
      }
      /// <summary>
      /// Restituisce il valore di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="columnName">Nome colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static T GetValue<T>(this IDataView dataView, string columnName, int row = 0)
      {
         var col = dataView.Schema.LastOrDefault(c => !c.IsHidden && c.Name == columnName);
         if (col.Name != columnName)
            return default;
         return dataView.GetValue<T>(col.Index, row);
      }
      /// <summary>
      /// Crea una IDataView che e' il merge dell'oggetto principale con una serie di altre IDataView
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="context">Contesto</param>
      /// <param name="others">Altre viste di dati da concatenare</param>
      /// <returns>La vista di dati concatenata</returns>
      public static IDataView Merge(this IDataView dataView, IMachineLearningContextProvider context, params IDataView[] others) =>
         DataViewMerged.Create(context, dataView.Schema, new[] { dataView }.Concat(others).ToArray());
      /// <summary>
      /// Equivalent to calling Equals(ColumnType) for non-vector types. For vector type,
      /// returns true if current and other vector types have the same size and item type.
      /// </summary>
      public static bool SameSizeAndItemType(this DataViewType columnType, DataViewType other)
      {
         if (other == null)
            return false;
         if (columnType.Equals(other))
            return true;
         // For vector types, we don't care about the factoring of the dimensions.
         if (columnType is not VectorDataViewType vectorType || other is not VectorDataViewType otherVectorType)
            return false;
         if (!vectorType.ItemType.Equals(otherVectorType.ItemType))
            return false;
         return vectorType.Size == otherVectorType.Size;
      }
      /// <summary>
      /// Trasforma la IDataView in una griglia di dati
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="context">Contesto</param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewGrid ToDataGrid(this IDataView dataView, IMachineLearningContextProvider context) => DataViewGrid.Create(context, dataView);
      /// <summary>
      /// Trasforma la IDataView in una data view filtrata
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="context">Contesto</param>
      /// <param name="filter">Filtro</param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewFiltered ToFiltered(this IDataView dataView, IMachineLearningContextProvider context, DataViewRowFilter filter) => DataViewFiltered.Create(context, dataView, filter);
      /// <summary>
      /// Converte una IDataView in un enumerable di coppie chiave/valore
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <returns>L'enumerable di coppie chiave/valore</returns>
      public static IEnumerable<KeyValuePair<string, object>[]> ToKeyValuePairs(this IDataView dataView)
      {
         foreach (var row in dataView.ToDataViewRows())
            yield return row.ToKeyValuePairs().ToArray();
      }
      /// <summary>
      /// Converte una DataViewRow in un enumerable di coppie chiave/valore
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <returns>L'enumerable di coppie chiave/valore</returns>
      public static IEnumerable<KeyValuePair<string, object>> ToKeyValuePairs(this DataViewRow row)
      {
         static object GetValue(DataViewRow row, DataViewSchema.Column col)
         {
            static T GetValue<T>(DataViewRow row, DataViewSchema.Column col)
            {
               var value = default(T);
               row.GetGetter<T>(col).Invoke(ref value);
               return value;
            }
            if (col.Type.RawType == typeof(float))
               return GetValue<float>(row, col);
            if (col.Type.RawType == typeof(ReadOnlyMemory<char>))
               return GetValue<ReadOnlyMemory<char>>(row, col).ToString();
            if (col.Type.RawType == typeof(VBuffer<float>))
               return GetValue<VBuffer<float>>(row, col).DenseValues().ToArray();
            if (col.Type.RawType == typeof(VBuffer<ReadOnlyMemory<char>>))
               return (from s in GetValue<VBuffer<ReadOnlyMemory<char>>>(row, col).DenseValues() select s.ToString()).ToArray();
            if (col.Type.RawType == typeof(DateTime))
               return GetValue<DateTime>(row, col);
            if (col.Type.RawType == typeof(bool))
               return GetValue<bool>(row, col);
            return null;
         }
         return from c in row.Schema where !c.IsHidden select new KeyValuePair<string, object>(c.Name, GetValue(row, c));
      }
      /// <summary>
      /// Converte una IDataView in un enumerable di coppie chiave/valore
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <returns>L'enumerable di coppie chiave/valore</returns>
      public static IEnumerable<DataViewRow> ToDataViewRows(this IDataView dataView)
      {
         var cursor = dataView.GetRowCursor(dataView.Schema);
         while (cursor.MoveNext())
            yield return cursor;
      }
      /// <summary>
      /// Converte una IDataView in un enumerable di IDataView per linea singola
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="ml">Contesto ml</param>
      /// <returns>L'enumerable di IDataView</returns>
      public static IEnumerable<IDataView> ToEnumerable(this IDataView dataView, MLContext ml)
      {
         while (dataView.GetRowCursor(dataView.Schema).MoveNext()) {
            yield return ml.Data.TakeRows(dataView, 1);
            dataView = ml.Data.SkipRows(dataView, 1);
         }
      }
      #endregion
   }
}
