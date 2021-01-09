using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning
{
   /// <summary>
   /// Estensioni per l'accesso ai gestori di dati
   /// </summary>
   public static class DataViewExtensions
   {
      #region Methods
      /// <summary>
      /// Trasforma un cursore in enumerabile
      /// </summary>
      /// <param name="cursor">Cursore di dato</param>
      /// <returns>L'enumerabile</returns>
      public static IEnumerable<DataViewRowCursor> AsEnumerable(this DataViewRowCursor cursor)
      {
         while (cursor.MoveNext())
            yield return cursor;
      }
      /// <summary>
      /// Restituisce un valore
      /// </summary>
      /// <param name="cursor">Il cursore</param>
      /// <param name="column">La colonna</param>
      /// <returns>Il valore</returns>
      public static DataValue GetValue(this DataViewRowCursor cursor, DataViewSchema.Column column) => GetValue(cursor, column.Index);
      /// <summary>
      /// Restituisce un valore
      /// </summary>
      /// <param name="cursor">Il cursore</param>
      /// <param name="columnIndex">L'indice di colonna</param>
      /// <returns>Il valore</returns>
      public static DataValue GetValue(this DataViewRowCursor cursor, int columnIndex)
      {
         // Crea il metodo generico del getter
         var getterMethodInfo = typeof(DataViewExtensions).GetMethod(nameof(GetValue), 1, new[] { typeof(DataViewRowCursor), typeof(int) });
         var getterGenericMethodInfo = getterMethodInfo.MakeGenericMethod(cursor.Schema[columnIndex].Type.RawType);
         // Lo invoca e restituisce il risultato
         return new DataValue(getterGenericMethodInfo.Invoke(null, new object[] { cursor, columnIndex }));
      }
      /// <summary>
      /// Restituisce un valore
      /// </summary>
      /// <param name="cursor">Il cursore</param>
      /// <param name="columnName">Il nome della colonna</param>
      /// <returns>Il valore</returns>
      public static DataValue GetValue(this DataViewRowCursor cursor, string columnName) => GetValue(cursor, cursor.Schema[columnName]);
      /// <summary>
      /// Restituisce un valore
      /// </summary>
      /// <typeparam name="T">Tipo di valore</typeparam>
      /// <param name="cursor">Il cursore</param>
      /// <param name="col">L'indice di colonna</param>
      /// <returns>Il valore</returns>
      public static T GetValue<T>(this DataViewRowCursor cursor, int col)
      {
         // Azione di restituzione dei valori
         var getter = cursor.GetGetter<T>(cursor.Schema[col]);
         T value = default;
         getter(ref value);
         return value;
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
      /// Trasforma la IDataView in una data view filtrata
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="context">Contesto</param>
      /// <param name="filter">Filtro</param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewFiltered ToDataViewFiltered(this IDataView dataView, IMachineLearningContextProvider context, DataViewRowFilter filter) => DataViewFiltered.Create(context, dataView, filter);
      /// <summary>
      /// Trasforma la IDataView in una griglia di dati
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="context">Contesto</param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewGrid ToDataViewGrid(this IDataView dataView, IMachineLearningContextProvider context) => DataViewGrid.Create(context, dataView);
      /// <summary>
      /// Trasforma un cursore di vista dati in una riga con valori
      /// </summary>
      /// <param name="cursor">Cursore</param>
      /// <param name="context">Contesto</param>
      /// <returns>La riga di dati</returns>
      public static DataViewValuesRow ToDataViewValuesRow(this DataViewRowCursor cursor, IMachineLearningContextProvider context) => DataViewValuesRow.Create(context, cursor);
      #endregion
   }
}
